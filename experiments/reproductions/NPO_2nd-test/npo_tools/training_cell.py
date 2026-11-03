def training_plan():
    steps = int(len(TRAIN_FORGET)*NOMINAL_EPOCHS/(TRAIN_GROUP*ACCUM_GROUPS))
    assert steps > WARMUP_STEPS >= 0 and len(TRAIN_FORGET) % TRAIN_GROUP == 0
    loader = DataLoader(list(range(len(TRAIN_FORGET))), batch_size=TRAIN_GROUP, shuffle=True,
                        generator=torch.Generator().manual_seed(SAMPLER_SEED), num_workers=0, drop_last=False)
    groups = []
    while len(groups) < steps*ACCUM_GROUPS:
        for group in loader:
            groups.append(group.tolist())
            if len(groups) == steps*ACCUM_GROUPS:
                break
    return steps, groups


def schedule_lr(completed_steps, total_steps):
    if completed_steps < WARMUP_STEPS:
        return LR*completed_steps/max(1, WARMUP_STEPS)
    return LR*max(0, (total_steps-completed_steps)/max(1, total_steps-WARMUP_STEPS))


def train_npo():
    receipt_path = NPO_DIR / "training_complete.json"
    if REUSE_NPO_PATH or (RESUME_RUN_PATH and receipt_path.is_file()):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["training_sha256"] == TRAIN_DIGEST, "Completed checkpoint has different training inputs/configuration."
        for name, sha in receipt["weight_hashes"].items():
            assert file_sha256(NPO_DIR/name) == sha, "Completed checkpoint bytes changed."
        print("Using verified completed NPO checkpoint:", NPO_DIR)
        return receipt
    if NPO_DIR.exists():
        assert RESUME_RUN_PATH, "Do not overwrite a checkpoint; create a fresh run."
        # Preserve an interrupted final export, then regenerate from the checkpoint.
        NPO_DIR.rename(NPO_DIR.with_name("npo_model_incomplete_" + str(time.time_ns())))
    cached = json.loads((RUN_DIR / "frozen_full_reference.json").read_text())
    assert cached["training_sha256"] == TRAIN_DIGEST and cached["reference_weights_sha256"] == MODEL_HASHES["full"]
    reference_nll = torch.tensor(cached["nll"], dtype=torch.float32, device=DEVICE)
    assert len(reference_nll) == len(TRAIN_FORGET) and not reference_nll.requires_grad
    steps, groups = training_plan()
    plan = {"steps": steps, "effective_batch": TRAIN_GROUP*ACCUM_GROUPS,
            "forget_groups": groups, "retain_pairs": RETAIN_PAIRS}
    identity = {"training_sha256": TRAIN_DIGEST, "plan_sha256": digest(plan),
                "reference_sha256": digest(cached), "packages": ENVIRONMENT["packages"]}
    durable_json(RUN_DIR / "training_plan.json", plan)
    release_memory()
    model = optimizer = None
    records = []
    try:
        model = load_model(FULL_DIR, train=True)
        assert all(p.requires_grad for p in model.parameters())
        assert not getattr(model, "is_quantized", False)
        assert model.config.attention_dropout == 0, "Serial/cached forwards assume zero attention dropout."
        model.eval()
        for i in (0, len(TRAIN_FORGET)//2, len(TRAIN_FORGET)-1):
            assert abs(score_one(model, TRAIN_FORGET[i])[0] - cached["nll"][i]) < 1e-3
        model.config.use_cache = False
        if GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        torch.manual_seed(SEED)
        checkpoint_root = RUN_DIR / "optimizer_checkpoints"
        print("Opening recoverable optimizer (initial state / checksum verification may take a few minutes)...", flush=True)
        optimizer = DiskMasterAdamW(model.named_parameters(), checkpoint_root,
            betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
            chunk_elements=CPU_CHUNK_ELEMENTS, resume=bool(RESUME_RUN_PATH and checkpoint_root.exists()),
            identity=identity, initial_metadata={"history": []}, buffers=model.named_buffers())
        if optimizer.step_count:
            records = optimizer.metadata["history"] + [optimizer.metadata["record"]]
            assert [r["update"] for r in records] == list(range(1, optimizer.step_count+1))
        assert optimizer.step_count <= steps
        optimizer.zero_grad()
        # Checkpoint metadata is authoritative; discard any uncommitted log tail.
        with (RUN_DIR / "training_log.jsonl").open("w", encoding="utf-8") as log:
            for record in records:
                log.write(json.dumps(record, allow_nan=False)+"\n")
            log.flush()
            progress = tqdm(range(optimizer.step_count, steps), initial=optimizer.step_count, total=steps,
                            desc=f"{OBJECTIVE}: recoverable updates")
            for step in progress:
                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                f_loss, r_loss = 0.0, 0.0
                progress.set_postfix(phase="forward/backward", refresh=True)
                for indices in groups[step*ACCUM_GROUPS:(step+1)*ACCUM_GROUPS]:
                    retain_ids = [RETAIN_PAIRS[i] for i in indices]
                    retain_tokens = sum(sum(v != -100 for v in TRAIN_RETAIN[j]["labels"][1:]) for j in retain_ids)
                    for i in indices:
                        current, _ = forward_nll(model, TRAIN_FORGET[i])
                        loss = npo_from_sequence_nll(current, reference_nll[i:i+1], BETA)
                        scaled = NPO_COEFF*loss/(len(indices)*ACCUM_GROUPS)
                        scaled.backward()
                        f_loss += float(loss.detach())/(len(indices)*ACCUM_GROUPS)
                        del current, loss, scaled
                    if RETAIN_COEFF:
                        for j in retain_ids:
                            nll, _ = forward_nll(model, TRAIN_RETAIN[j])
                            retain_term = nll.sum()/retain_tokens
                            (RETAIN_COEFF*retain_term/ACCUM_GROUPS).backward()
                            r_loss += float(retain_term.detach())/ACCUM_GROUPS
                            del nll, retain_term
                torch.cuda.synchronize()
                compute_seconds = time.perf_counter()-start
                lr = schedule_lr(step, steps)
                record = {"update": step+1, "lr": lr, "npo_loss": f_loss, "retain_ce": r_loss,
                    "total_loss": NPO_COEFF*f_loss+RETAIN_COEFF*r_loss,
                    "compute_seconds": compute_seconds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
                    "process_ram_gib": psutil.Process().memory_info().rss/1024**3}
                assert all(math.isfinite(v) for v in record.values())
                progress.set_postfix(phase="AdamW + durable checkpoint", refresh=True)
                opt_start = time.perf_counter()
                previous_threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(OPTIMIZER_CPU_THREADS)
                    optimizer.step(lr, metadata={"history": records, "record": record})
                finally:
                    torch.set_num_threads(previous_threads)
                optimizer.zero_grad()
                record = dict(optimizer.metadata["record"])
                record.update(optimizer_and_checkpoint_seconds=time.perf_counter()-opt_start,
                              seconds=time.perf_counter()-start)
                records.append(record)
                log.write(json.dumps(record, allow_nan=False)+"\n")
                log.flush()
                progress.set_postfix(npo=f"{f_loss:.3f}", compute=f"{compute_seconds:.1f}s",
                    optimizer=f"{record['optimizer_and_checkpoint_seconds']:.1f}s", saved=step+1)
        if GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        model.to("cpu")
        release_memory()
        model.save_pretrained(str(NPO_DIR), safe_serialization=True, max_shard_size="1GB")
        tokenizer.save_pretrained(str(NPO_DIR))
        # Flush exported weights before publishing their completion receipt.
        for path in NPO_DIR.iterdir():
            if path.is_file():
                with path.open("r+b") as handle:
                    os.fsync(handle.fileno())
        weight_hashes = {path.name: file_sha256(path) for path in sorted(NPO_DIR.glob("*.safetensors"))}
        assert weight_hashes
        receipt = {"status": "complete", "training_sha256": TRAIN_DIGEST, "training": TRAIN_IDENTITY,
                   "updates": steps, "weight_hashes": weight_hashes, "quantized_during_training": False}
        durable_json(NPO_DIR / "training_complete.json", receipt)
        pd.DataFrame(records).to_csv(RUN_DIR / "training_log.csv", index=False)
        print("Saved complete NPO model:", NPO_DIR)
        print("Recovery checkpoints retained:", checkpoint_root)
        return receipt
    except (Exception, KeyboardInterrupt) as exc:
        durable_json(RUN_DIR / "training_failure.json", {"error": repr(exc),
            "last_committed_update": optimizer.step_count if optimizer is not None else None,
            "instruction": "Restart kernel, set RESUME_RUN_PATH to this run folder, and run all cells. "
                           "Resume verifies saved generations; a partially written update is replayed."})
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad()
        del optimizer, model, reference_nll
        release_memory()


training_receipt = train_npo()
if (RUN_DIR / "training_log.csv").is_file():
    history = pd.read_csv(RUN_DIR / "training_log.csv")
    display(history.tail())
    history.plot(x="update", y=["npo_loss", "retain_ce"], title=f"{OBJECTIVE}: measured training losses")
    plt.tight_layout()
    plt.show()
