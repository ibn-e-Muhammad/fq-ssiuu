"""Small local optimizer timing probe; no user model or run is loaded."""
from validate_recovery import *
import statistics

torch.set_num_threads(1)
measurements = []
with tempfile.TemporaryDirectory(dir=ROOT) as folder:
    # 16 MiB per FP32 array, same 4 MiB chunk size as real training.
    for threads in (8, 1, 2, 4):
        torch.set_num_threads(threads)
        p = torch.nn.Parameter(torch.zeros((2048, 2048), device="cuda", dtype=torch.bfloat16))
        opt = DiskMasterAdamW([("weight", p)], Path(folder) / str(threads))
        samples = []
        for step in range(3):
            p.grad = torch.full_like(p, 0.01)
            torch.cuda.synchronize()
            start = time.perf_counter()
            opt.step(0.001)
            torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
        measurements.append({"cpu_threads": threads, "seconds": samples, "median_seconds": statistics.median(samples)})
        del p, opt
print(json.dumps(measurements, indent=2))
(ROOT / "npo_tools/optimizer_benchmark.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
