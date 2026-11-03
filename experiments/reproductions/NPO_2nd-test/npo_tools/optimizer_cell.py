def durable_json(path, value):
    """Publish metadata only after its bytes have been flushed to the device."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class DiskMasterAdamW:
    """Immutable, checksummed FP32 AdamW generations; commit once per update.

    A failed update never changes the previous committed generation. BF16 model
    weights are reconstructed by casting the saved FP32 masters on resume.
    """
    FILES = ("master.bin", "moment1.bin", "moment2.bin", "runtime.pt", "commit.json", "commit.json.tmp")

    def __init__(self, named_parameters, directory, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, chunk_elements=1_048_576, resume=False,
                 identity=None, initial_metadata=None, buffers=None):
        self.params = [(name, p) for name, p in named_parameters if p.requires_grad]
        assert self.params and len({id(p) for _, p in self.params}) == len(self.params)
        assert all(p.is_contiguous() for _, p in self.params)
        self.directory = Path(directory)
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        self.chunk_elements = chunk_elements
        assert chunk_elements > 0
        self.buffers = list(buffers or [])
        self.total_elements = sum(p.numel() for _, p in self.params)
        self.contract = {"version": 2, "identity": identity,
            "parameters": [[n, list(p.shape), str(p.dtype)] for n, p in self.params],
            "betas": list(betas), "eps": eps, "weight_decay": weight_decay}
        self.step_count = 0
        self.metadata = initial_metadata or {}
        self.current = None
        if resume:
            if not self.directory.is_dir():
                raise ValueError("No recoverable optimizer directory. Legacy scratch files cannot be resumed.")
            self._restore()
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._require_space(3)  # two retained checkpoints plus the next candidate
            candidate = self._candidate(0)
            hashes = {k: hashlib.sha256() for k in ("master", "moment1", "moment2")}
            with ExitStack() as stack:
                files = {k: stack.enter_context((candidate / (k + ".bin")).open("xb")) for k in hashes}
                for _, param in self.params:
                    flat = param.detach().view(-1)
                    for start in range(0, flat.numel(), self.chunk_elements):
                        chunk = flat[start:start+self.chunk_elements].to(device="cpu", dtype=torch.float32)
                        self._append(files["master"], hashes["master"], chunk)
                        zeros = bytes(chunk.numel()*4)
                        for key in ("moment1", "moment2"):
                            files[key].write(zeros)
                            hashes[key].update(zeros)
                self._flush(files)
            self._commit(candidate, hashes, self.metadata, 0)

    def _require_space(self, copies=1):
        required = self.total_elements * 12 * copies + 2 * 1024**3
        if shutil.disk_usage(self.directory).free < required:
            raise OSError(f"Need {required/1024**3:.1f} GiB free for recoverable optimizer files.")

    def _candidate(self, step):
        import uuid
        path = self.directory / f"step-{step:06d}-{uuid.uuid4().hex}"
        path.mkdir(exist_ok=False)
        return path

    def _generations(self):
        return sorted((p for p in self.directory.iterdir()
            if p.is_dir() and re.fullmatch(r"step-\d{6}-[0-9a-f]{32}", p.name)), reverse=True)

    def _remove_generation(self, path):
        # Delete only this optimizer's known files inside its immediate child.
        if path.is_symlink() or path.resolve().parent != self.directory.resolve():
            raise ValueError("Checkpoint cleanup escaped the optimizer directory.")
        for name in self.FILES:
            (path / name).unlink(missing_ok=True)
        path.rmdir()  # unexpected files are never recursively deleted

    def _prune(self):
        older = [p for p in self._generations() if p < self.current and (p / "commit.json").is_file()]
        keep = {self.current, *older[:1]}
        for path in self._generations():
            if path not in keep:
                self._remove_generation(path)

    @staticmethod
    def _flush(files):
        for handle in files.values():
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _append(handle, checksum, value):
        raw = value.numpy().tobytes()
        handle.write(raw)
        checksum.update(raw)

    @staticmethod
    def _read(handle, count):
        raw = handle.read(count*4)
        if len(raw) != count*4:
            raise IOError("Truncated optimizer state.")
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())

    def _runtime(self):
        ns = np.random.get_state()
        return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
            "buffers": {n: b.detach().cpu().clone() for n, b in self.buffers}}

    def _commit(self, candidate, hashes, metadata, step):
        runtime_path = candidate / "runtime.pt"
        with runtime_path.open("xb") as handle:
            torch.save(self._runtime(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        files = {k + ".bin": {"bytes": 4*self.total_elements, "sha256": h.hexdigest()} for k, h in hashes.items()}
        files["runtime.pt"] = {"bytes": runtime_path.stat().st_size, "sha256": file_sha256(runtime_path)}
        # The commit marker is last. Incomplete directories are ignored on resume.
        durable_json(candidate / "commit.json", {"contract": self.contract,
            "completed_steps": step, "metadata": metadata, "files": files})
        self.current, self.step_count, self.metadata = candidate, step, metadata
        self.paths = {k: candidate / (k + ".bin") for k in hashes}
        self._prune()

    @torch.no_grad()
    def _restore(self):
        errors = []
        selected = None
        for path in self._generations():
            try:
                manifest = json.loads((path / "commit.json").read_text(encoding="utf-8"))
                if manifest["contract"] != self.contract:
                    raise ValueError("Training configuration/model layout differs from checkpoint.")
                assert manifest["completed_steps"] == int(path.name.split("-")[1])
                assert set(manifest["files"]) == {"master.bin", "moment1.bin", "moment2.bin", "runtime.pt"}
                for name, info in manifest["files"].items():
                    item = path / name
                    assert item.stat().st_size == info["bytes"], f"Size mismatch: {name}"
                    if name.endswith(".bin"):
                        assert info["bytes"] == self.total_elements*4
                    assert file_sha256(item) == info["sha256"], f"Checksum mismatch: {name}"
                runtime = torch.load(path / "runtime.pt", map_location="cpu", weights_only=True)
                assert set(runtime["buffers"]) == {n for n, _ in self.buffers}
                for name, buf in self.buffers:
                    assert runtime["buffers"][name].shape == buf.shape
                selected = (path, manifest, runtime)
                break
            except (OSError, ValueError, KeyError, AssertionError, RuntimeError, EOFError) as exc:
                errors.append(f"{path.name}: {exc}")
        if selected is None:
            raise ValueError("No valid compatible checkpoint; legacy scratch cannot resume. " + "; ".join(errors))
        path, manifest, runtime = selected
        if errors:
            print("Skipped incomplete/corrupt checkpoints:", *errors, sep="\n")
        self.current, self.step_count = path, manifest["completed_steps"]
        self.metadata = manifest["metadata"]
        self.paths = {k: path / (k + ".bin") for k in ("master", "moment1", "moment2")}
        with self.paths["master"].open("rb") as handle:
            for _, param in self.params:
                flat = param.view(-1)
                for start in range(0, flat.numel(), self.chunk_elements):
                    count = min(self.chunk_elements, flat.numel()-start)
                    flat[start:start+count].copy_(self._read(handle, count))
        for name, buf in self.buffers:
            buf.copy_(runtime["buffers"][name])
        random.setstate(runtime["python"])
        ns = runtime["numpy"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(runtime["torch_cpu"])
        if runtime["torch_cuda"]:
            torch.cuda.set_rng_state_all(runtime["torch_cuda"])
        self._prune()
        print(f"Verified checkpoint: {path.name}; next update: {self.step_count+1}")

    def zero_grad(self):
        for _, p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, lr, metadata=None):
        started = time.perf_counter()
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("Invalid learning rate.")
        # Accumulate finite flags on device; synchronize once per parameter,
        # instead of once per chunk (over 1,000 synchronizations for this model).
        for name, p in self.params:
            if p.grad is None or p.grad.is_sparse:
                raise ValueError(f"Missing/dense gradient required: {name}")
            flat = p.grad.detach().view(-1)
            finite = torch.ones((), dtype=torch.bool, device=flat.device)
            for start in range(0, flat.numel(), self.chunk_elements):
                finite.logical_and_(torch.isfinite(flat[start:start+self.chunk_elements]).all())
            if not bool(finite):
                raise FloatingPointError(f"Nonfinite gradient: {name}; checkpoint unchanged.")
        self._require_space()
        candidate = self._candidate(self.step_count+1)
        hashes = {k: hashlib.sha256() for k in self.paths}
        b1, b2 = self.betas
        t = self.step_count+1
        bc1, bc2 = 1-b1**t, 1-b2**t
        with ExitStack() as stack:
            source = {k: stack.enter_context(path.open("rb")) for k, path in self.paths.items()}
            target = {k: stack.enter_context((candidate / (k + ".bin")).open("xb")) for k in source}
            for name, p in self.params:
                flat, grads = p.view(-1), p.grad.detach().view(-1)
                decay = self.weight_decay if p.ndim >= 2 and not name.endswith("bias") else 0.0
                for start in range(0, flat.numel(), self.chunk_elements):
                    count = min(self.chunk_elements, flat.numel()-start)
                    master, m, v = (self._read(source[k], count) for k in ("master", "moment1", "moment2"))
                    grad = grads[start:start+count].to(device="cpu", dtype=torch.float32)
                    master.mul_(1-lr*decay)
                    m.lerp_(grad, 1-b1)
                    v.mul_(b2).addcmul_(grad, grad, value=1-b2)
                    denom = (v.sqrt()/math.sqrt(bc2)).add_(self.eps)
                    master.addcdiv_(m, denom, value=-lr/bc1)
                    if not torch.isfinite(master).all():
                        raise FloatingPointError("Nonfinite AdamW master; resume last committed checkpoint.")
                    for key, value in (("master", master), ("moment1", m), ("moment2", v)):
                        self._append(target[key], hashes[key], value)
                    flat[start:start+count].copy_(master)
            self._flush(target)
        metadata = dict(metadata or {})
        if "record" in metadata:
            metadata["record"] = dict(metadata["record"], optimizer_seconds=time.perf_counter()-started)
            metadata["record"]["seconds"] = (metadata["record"].get("compute_seconds", 0)
                                               + metadata["record"]["optimizer_seconds"])
        self._commit(candidate, hashes, metadata, t)

    def remove_completed_scratch(self):
        # Recovery checkpoints intentionally survive final model export.
        print("Retained the latest two recovery checkpoints:", self.directory)
