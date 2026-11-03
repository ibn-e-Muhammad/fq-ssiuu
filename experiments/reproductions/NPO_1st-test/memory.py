"""Selected-device memory evidence, including memory outside PyTorch's allocator."""

import json
import threading
import time
from pathlib import Path

import torch


class MemoryMonitor:
    def __init__(self, path, limit=16000000000, device_index=0):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.uuid = pynvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(self.uuid, bytes):
            self.uuid = self.uuid.decode("ascii")
        self.limit = limit
        self.peak_device_used = 0
        self.errors = []
        self.stop = threading.Event()
        self.log = Path(path).open("x")
        torch.cuda.reset_peak_memory_stats(0)
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        try:
            while not self.stop.is_set():
                used = int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
                self.peak_device_used = max(self.peak_device_used, used)
                self.log.write(
                    json.dumps({"monotonic_seconds": time.monotonic(), "device_used_bytes": used})
                    + "\n"
                )
                self.log.flush()
                self.stop.wait(0.1)
        except Exception as error:
            self.errors.append(type(error).__name__)

    def snapshot(self):
        torch.cuda.synchronize(0)
        try:
            current = int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
        except Exception as error:
            if self.limit is not None:
                raise
            self.errors.append(type(error).__name__)
            current = 0
        self.peak_device_used = max(current, self.peak_device_used)
        result = {
            "device_uuid": self.uuid,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "sampled_peak_device_used_bytes": self.peak_device_used,
            "device_sampling_interval_seconds": 0.1,
            "qualification": "NVML sampled device peak includes managed/non-PyTorch memory; sub-interval transients are not continuously measured.",
            "limit_bytes": self.limit,
        }
        result["telemetry_errors"] = list(self.errors)
        return result

    def observe(self):
        result = self.snapshot()
        if self.errors and self.limit is not None:
            raise RuntimeError("Device memory telemetry failed")
        if self.limit is not None and (
            max(
                result[k]
                for k in (
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "sampled_peak_device_used_bytes",
                )
            )
            >= self.limit
        ):
            raise MemoryError("Selected-device memory budget exceeded")
        return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        self.log.close()
        self.nvml.nvmlShutdown()
