"""Build the self-contained notebook cells; preserves all unaffected cells."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = root / "NPO_LOCALL_NOTE.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))


def source(i):
    return "".join(notebook["cells"][i]["source"])


def put(i, value):
    cell = notebook["cells"][i]
    cell["source"] = value.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


assert "RESUME_RUN_PATH =" not in source(3), "Notebook already upgraded."
s = source(3).replace("CPU_CHUNK_ELEMENTS = 1_048_576  # 4 MiB per FP32 chunk; bounded optimizer workspace",
    "CPU_CHUNK_ELEMENTS = 1_048_576  # try 4_194_304 after measuring optimizer time\n"
    "OPTIMIZER_CPU_THREADS = 4     # small local timing probe favored 4; full-run timing may differ\n"
    "GRADIENT_CHECKPOINTING = True  # keep enabled until VRAM headroom is measured")
s = s.replace("# Optional reuse of a COMPLETED checkpoint from this notebook. No partial resume.",
    "# Recovery: use the RUN FOLDER printed by a previous run of this updated notebook.\n"
    'RESUME_RUN_PATH = None    # e.g. r"C:\\FYP\\npo_results\\20261030_..."\n'
    "# Optional reuse of a COMPLETED model, mutually exclusive with resume.")
s = s.replace("KEEP_OPTIMIZER_FILES = False  # remove only this run's 3 scratch files after a successful save",
    "# Latest two recovery checkpoints are retained even after successful model export.")
s = s.replace('RUN_DIR = RUN_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}")\nRUN_DIR.mkdir(parents=True, exist_ok=False)',
    'assert not (RESUME_RUN_PATH and REUSE_NPO_PATH), "Choose resume OR completed-model reuse."\n'
    'if RESUME_RUN_PATH:\n'
    '    RUN_DIR = Path(RESUME_RUN_PATH).resolve()\n'
    '    assert (RUN_DIR / "manifest.json").is_file(), "Resume requires an existing run manifest."\n'
    '    assert (RUN_DIR / "manifest.json").read_text(encoding="utf-8").find("transactional-disk-FP32-master-AdamW-v2") >= 0, "Legacy interrupted runs cannot be resumed; start a fresh run."\n'
    'else:\n'
    '    RUN_DIR = RUN_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}")\n'
    '    RUN_DIR.mkdir(parents=True, exist_ok=False)')
s += '\nassert OPTIMIZER_CPU_THREADS >= 1\nprint("Resume run:", RESUME_RUN_PATH or "No (fresh experiment)")\n'
put(3, s)
s = source(4).replace('assert REUSE_NPO_PATH or free_disk >= 28 * 1024**3, "Choose an SSD output folder with at least 28 GiB free."',
    'assert REUSE_NPO_PATH or RESUME_RUN_PATH or free_disk >= 45 * 1024**3, "Recoverable optimizer needs at least 45 GiB free on a local SSD for this model."')
put(4, s)
s = source(7).replace('"streaming-disk-FP32-master-AdamW-v1"', '"transactional-disk-FP32-master-AdamW-v2"')
s = s.replace('"warmup_steps": WARMUP_STEPS,', '"gradient_checkpointing": GRADIENT_CHECKPOINTING, "warmup_steps": WARMUP_STEPS,')
s = s.replace('json_write(RUN_DIR / "manifest.json",',
    'if RESUME_RUN_PATH:\n'
    '    previous_manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))\n'
    '    assert previous_manifest["training_sha256"] == TRAIN_DIGEST, "Resume training settings/data changed."\n'
    '    assert previous_manifest["evaluation_sha256"] == EVAL_DIGEST, "Resume evaluation settings changed."\n'
    'json_write(RUN_DIR / ("resume_manifest.json" if RESUME_RUN_PATH else "manifest.json"),')
put(7, s)
s = source(11)
s = s.replace('    stage_dir.mkdir(exist_ok=False)', '''    if stage_dir.exists():
        meta_path = stage_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if meta.get("status") == "complete":
            assert meta["evaluation_sha256"] == EVAL_DIGEST and meta["model_path"] == str(directory)
            assert meta["quantization"] == quant
            for name, sha in meta["artifact_hashes"].items():
                assert file_sha256(stage_dir / name) == sha, "Cached evaluation changed."
            if cache_reference:
                reference = json.loads((RUN_DIR / "frozen_full_reference.json").read_text())
                assert reference["training_sha256"] == TRAIN_DIGEST
                assert reference["reference_weights_sha256"] == MODEL_HASHES["full"]
                assert digest(reference) == meta["reference_sha256"]
            frame = pd.read_json(stage_dir / "per_question.jsonl", lines=True)
            STAGE_RESULTS[stage], STAGE_META[stage] = frame, meta
            print("Reusing verified completed evaluation:", stage)
            return frame
        # Preserve partial outputs; reevaluate only this interrupted stage.
        stage_dir.rename(stage_dir.with_name(stage + "_incomplete_" + str(time.time_ns())))
    stage_dir.mkdir(exist_ok=False)''')
s = s.replace('        meta.update(status="complete",', '''        meta["artifact_hashes"] = {name: file_sha256(stage_dir / name)
            for name in ("per_question.jsonl", "per_question.csv", "metrics.csv")}
        if cache_reference:
            meta["reference_sha256"] = digest(json.loads((RUN_DIR / "frozen_full_reference.json").read_text()))
        meta.update(status="complete",''')
put(11, s)
original_selftest = source(16)[source(16).index("def optimizer_self_test():"):]
put(16, (root / "npo_tools/optimizer_cell.py").read_text(encoding="utf-8") + "\n\n" + original_selftest)
put(17, (root / "npo_tools/training_cell.py").read_text(encoding="utf-8"))
s = source(0)
a = s.index('Keep **30 GB')
b = s.index('\n\nThis is a local', a)
s = s[:a] + ('Keep **at least 45 GiB free on a local SSD** in addition to downloaded models. '
    'The latest two optimizer generations are retained; a third exists while an update is written '
    '(about 42 GiB peak optimizer storage). Every completed update is a checksummed recovery checkpoint, '
    'including FP32 masters/moments, training position, and random state. After interruption set '
    '`RESUME_RUN_PATH` to the same run folder and run all cells. Completed evaluations are verified and reused. '
    'Old interrupted runs made before this checkpoint implementation cannot be recovered.') + s[b:]
put(0, s)
s = source(15)
a = s.index('**Expect slow SSD-heavy updates.**')
s = s[:a] + '''**Recovery after every update:** the optimizer streams new values to a new generation instead of overwriting the last checkpoint. It flushes state files, saves RNG/model buffers and training metadata, then publishes a commit marker. The latest two committed generations are kept; incomplete/corrupt generations are skipped after checksum verification. This uses the optimizer's existing read/write pass, avoiding a separate 15 GB checkpoint copy. Checksums and durable flushes still add overhead. Recovery depends on the filesystem/drive honoring flushes; a UPS also protects against hardware-level write loss.

**Resume:** restart the kernel, set `RESUME_RUN_PATH` in the configuration cell to this run's folder, keep the scientific settings unchanged, and run all cells. The model's BF16 parameters are reconstructed from the FP32 masters, Adam moments and update count are restored, and the deterministic sample schedule continues at the next update. Completed evaluations are reused. An interrupted optimizer update is replayed. Never point resume at the old `optimizer_scratch` files. No checkpoint is selected by its forget score.

**Speed controls:** progress shows forward/backward versus optimizer/checkpoint phases; `training_log.jsonl` records their times. The optimizer now uses sequential files and fewer GPU synchronization points. `OPTIMIZER_CPU_THREADS=4` is a starting point from a small, noisy local timing probe; compare 1/2/4/8 using full-update timings. It is not a measured full-model speedup. You may change that or `CPU_CHUNK_ELEMENTS` when resuming (these do not change the objective). Larger chunks such as 4,194,304 use more working RAM and may reduce Python/I/O-call overhead. No specific speedup is guaranteed. Keep `GRADIENT_CHECKPOINTING=True` initially; switching it off trades more VRAM for less recomputation and requires a fresh run. Do not reduce the effective batch or epochs to claim the same experiment.
'''
put(15, s)
s = source(28)
s = s.replace('**Rerun:** restart the kernel for a fresh experiment directory.',
    '**Rerun/resume:** restart the kernel. Leave `RESUME_RUN_PATH=None` for a fresh experiment or set it to the interrupted run folder to restore its latest valid checkpoint and reuse completed evaluations.')
s = s.replace('Disk full or interruption → preserve completed evaluation files, discard the incomplete training run, free SSD space, and start a new run.',
    'Disk full or interruption → free SSD space, restart the kernel and set `RESUME_RUN_PATH` to the same run folder. Checkpoints from the old non-resumable implementation cannot be used.')
s += '\n\nRecovery implementation validation is in `npo_tools/validate_recovery.py`; it uses tiny tensors and injected failures, not a real power-cut or a full Llama training run.\n'
put(28, s)
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=f"cell_{i}")
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Updated notebook; all code cells parse.")
