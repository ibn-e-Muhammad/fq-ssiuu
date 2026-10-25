# Embedded verbatim in Mother_Local_v2.ipynb; the notebook needs no script imports.
import os
import shutil
import uuid


def checkpoint_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def find_sft_checkpoint(location):
    """Accept a run, its sft_checkpoints folder, or one explicit checkpoint."""
    location = Path(location).resolve()
    explicit = location.name.startswith('step-')
    folder = location/'sft_checkpoints' if (location/'sft_checkpoints').is_dir() else location
    candidates = [location] if explicit else sorted(folder.glob('step-*'), reverse=True)
    for candidate in candidates:
        try:
            manifest = json.loads((candidate/'complete.json').read_text(encoding='utf-8'))
            required = {'adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'training_state.pt'}
            if not required.issubset(manifest['sha256']):
                raise ValueError('Missing required checkpoint files')
            for name, digest in manifest['sha256'].items():
                path = (candidate/name).resolve()
                if not path.is_relative_to(candidate.resolve()) or checkpoint_digest(path) != digest:
                    raise ValueError(f'Checkpoint integrity check failed: {name}')
            return candidate
        except (OSError, ValueError, KeyError, TypeError) as error:
            print('Skipping invalid checkpoint:', candidate, '|', error)
    raise FileNotFoundError(f'No complete, intact SFT checkpoint found in {location}')


def save_sft_checkpoint(state, policy, tokenizer):
    """Publish a new directory only after every file has been flushed and hashed."""
    root = (EXPERIMENT_DIR/'sft_checkpoints').resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root/f"step-{state['step']:08d}"
    if destination.exists():
        raise FileExistsError(f'Checkpoint already exists: {destination}; use a new run')
    temporary = root/f'.incomplete-{uuid.uuid4().hex}'
    temporary.mkdir()
    policy.save_pretrained(str(temporary/'adapter'), safe_serialization=True)
    tokenizer.save_pretrained(str(temporary/'adapter'))
    torch.save(state, temporary/'training_state.pt')
    digests = {}
    for path in sorted(temporary.rglob('*')):
        if path.is_file():
            with path.open('rb+') as stream:
                os.fsync(stream.fileno())
            digests[path.relative_to(temporary).as_posix()] = checkpoint_digest(path)
    with (temporary/'complete.json').open('w', encoding='utf-8') as stream:
        json.dump({'step': state['step'], 'sha256': digests}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(destination)
    print('Saved recovery checkpoint:', destination)
    # Retain two committed saves by default. Never prune the source resume run.
    previous = sorted(root.glob('step-*'), reverse=True)
    for old in previous[SFT_KEEP_CHECKPOINTS:]:
        resolved = old.resolve()
        if resolved.parent != root or old.is_symlink() or not (old/'complete.json').is_file():
            raise RuntimeError(f'Refusing to prune unexpected checkpoint path: {old}')
        shutil.rmtree(resolved)
    return destination


if REUSE_SFT_MODEL is not None:
    training_record = json.loads((SFT_MODEL_DIR/'sft_training.json').read_text(encoding='utf-8'))
    assert training_record['dataset_sha256'] == full_dataset_sha256, 'Saved SFT used different TOFU rows'
    assert training_record['completed_full_epochs'] >= 1, 'This is an incomplete/smoke checkpoint; complete SFT first'
    assert training_record['max_length'] == MAX_LEN, 'Match MAX_LEN to the saved SFT training settings'
    print('Reusing complete SFT checkpoint:', SFT_MODEL_DIR)
else:
    if SFT_MODEL_DIR.exists() or SFT_ADAPTER.exists():
        raise FileExistsError('SFT output already exists. Rerun configuration for a new run; optionally set SFT_RESUME_FROM.')
    assert SFT_SAVE_EVERY > 0 and SFT_KEEP_CHECKPOINTS >= 2
    total_steps = SFT_EPOCHS * math.ceil(len(sft_rows)/GRAD_ACCUM)
    target_steps = min(total_steps, SFT_MAX_STEPS) if SFT_MAX_STEPS is not None else total_steps
    # Encoded-data identity also detects tokenizer, template and truncation changes.
    signature = dict(version=1, dataset_sha256=full_dataset_sha256,
        encoded_sha256=hashlib.sha256(json.dumps(sft_rows, sort_keys=True).encode()).hexdigest(),
        model_path=str(MODEL_PATH.resolve()), max_length=MAX_LEN, seed=SEED,
        epochs=SFT_EPOCHS, max_steps=SFT_MAX_STEPS, accumulation=GRAD_ACCUM, lr=LR_SFT,
        lora=json.loads(json.dumps(lora_config.to_dict(), default=lambda x: sorted(x) if isinstance(x, set) else str(x))),
        schedule='cosine_5pct_warmup', total_steps=total_steps)
    resume_state = None
    resumed_from = None
    if SFT_RESUME_FROM is not None:
        resumed_from = find_sft_checkpoint(SFT_RESUME_FROM)
        # This stores only tensors and basic Python data; never unpickle arbitrary objects.
        resume_state = torch.load(resumed_from/'training_state.pt', map_location='cpu', weights_only=True)
        if resume_state['signature'] != signature:
            changed = [key for key in signature if resume_state['signature'].get(key) != signature[key]]
            raise ValueError(f'Resume settings/data differ: {changed}. Restore the original settings or start fresh.')
        policy = PeftModel.from_pretrained(load_fp16(MODEL_PATH), str(resumed_from/'adapter'), is_trainable=True)
    else:
        policy = get_peft_model(load_fp16(MODEL_PATH), lora_config)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    policy.enable_input_require_grads()
    policy.config.use_cache = False
    policy.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR_SFT)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05*total_steps), total_steps)
    rng = random.Random(SEED)
    step = 0; examples_seen = 0; completed_full_epochs = 0; sft_history = []
    epoch = 0; position = 0; order = None
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        step = resume_state['step']; examples_seen = resume_state['examples_seen']
        completed_full_epochs = resume_state['completed_full_epochs']
        epoch = resume_state['epoch']; position = resume_state['position']; order = resume_state['order']
        sft_history = resume_state['history']
        rng.setstate(resume_state['shuffle_rng'])
        random.setstate(resume_state['python_rng'])
        nr = resume_state['numpy_rng']
        np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        torch.set_rng_state(resume_state['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_state['cuda_rng'])
        if step < target_steps:
            print(f'Resuming after optimizer step {step}; next step {step+1}, epoch {epoch+1}')
        else:
            print(f'All {step} optimizer steps are saved; finishing model export.')
        # Give the new run its own recovery point even before its next 50-step save.
        save_sft_checkpoint(resume_state, policy, tokenizer)
        del resume_state
    policy.train()
    while epoch < SFT_EPOCHS and step < target_steps:
        if order is None:
            order = list(range(len(sft_rows))); rng.shuffle(order)
        indices = order[position:position+GRAD_ACCUM]
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for index in indices:
            batch = collate([sft_rows[index]])
            logps, token_count = answer_logps(policy, batch)
            loss = -(logps/token_count).mean()/len(indices)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite SFT loss')
            loss.backward()
            mean_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        step += 1; position += len(indices); examples_seen += len(indices)
        if step == 1 or step % 10 == 0:
            sft_history.append({'step': step, 'epoch': epoch+1, 'loss_per_token': mean_loss})
            print(f'SFT {step}/{total_steps}, epoch {epoch+1}, loss/token {mean_loss:.4f}')
        if position == len(sft_rows):
            completed_full_epochs += 1; epoch += 1; position = 0; order = None
        # Save only at a completed optimizer update, including the final partial batch.
        if step % SFT_SAVE_EVERY == 0 or step == target_steps:
            nr = np.random.get_state()
            state = dict(signature=signature, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                step=step, epoch=epoch, position=position, order=order, examples_seen=examples_seen,
                completed_full_epochs=completed_full_epochs, history=sft_history,
                shuffle_rng=rng.getstate(), python_rng=random.getstate(),
                numpy_rng=(nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            save_sft_checkpoint(state, policy, tokenizer)
            del state
    if SFT_MAX_STEPS is None:
        assert completed_full_epochs == SFT_EPOCHS
        assert examples_seen == SFT_EPOCHS*len(sft_rows)
    policy.save_pretrained(str(SFT_ADAPTER))
    tokenizer.save_pretrained(str(SFT_ADAPTER))
    (EXPERIMENT_DIR/'sft_history.json').write_text(json.dumps(sft_history, indent=2))
    # The scheduler holds the optimizer; release both before loading the merge base.
    del scheduler, optimizer, policy
    for name in ('loss', 'logps', 'token_count', 'batch'):
        globals().pop(name, None)
    gc.collect(); torch.cuda.empty_cache()

    base_cpu = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.float16,
        device_map='cpu', local_files_only=True)
    sft_merged = PeftModel.from_pretrained(base_cpu, str(SFT_ADAPTER), is_trainable=False).merge_and_unload(safe_merge=True)
    sft_merged.config.use_cache = True
    sft_merged.save_pretrained(str(SFT_MODEL_DIR), safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(str(SFT_MODEL_DIR))
    training_record = dict(dataset='locuslab/TOFU', split='full', dataset_rows=len(sft_rows),
        dataset_sha256=full_dataset_sha256, examples_seen=examples_seen,
        completed_full_epochs=completed_full_epochs, optimizer_steps=step,
        max_length=MAX_LEN, seed=SEED, base_model=str(MODEL_PATH),
        checkpoint_type='merged_full_model_fp16', smoke_only=bool(options.get('smoke')),
        resumed_from=str(resumed_from) if resumed_from else None)
    (SFT_MODEL_DIR/'sft_training.json').write_text(json.dumps(training_record, indent=2), encoding='utf-8')
    del base_cpu, sft_merged; gc.collect()
    print('Saved complete fine-tuned model:', SFT_MODEL_DIR)
    print('Full epochs completed:', completed_full_epochs, '| Examples seen:', examples_seen)
