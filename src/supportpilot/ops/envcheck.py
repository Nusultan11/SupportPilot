"""Hardware / CUDA environment report (`supportpilot env`)."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from pydantic import BaseModel

GIB = 1024**3


class GpuInfo(BaseModel):
    index: int
    name: str
    capability: str
    total_vram_gib: float
    free_vram_gib: float
    bf16_supported: bool
    matmul_ok: bool | None = None
    matmul_tflops_fp16: float | None = None


class EnvReport(BaseModel):
    created_at: datetime
    python: str
    platform: str
    cpu: str
    physical_cores: int | None
    logical_cores: int | None
    ram_total_gib: float
    ram_available_gib: float
    disk_free_gib: float
    nvidia_driver: str | None
    torch: str | None
    torch_cuda: str | None
    cuda_available: bool
    cudnn: int | None
    gpus: list[GpuInfo]
    problems: list[str]


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _nvidia_driver() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None


def _matmul_benchmark(device_index: int) -> tuple[bool, float]:
    """Run a 2048² fp16 matmul a few times; returns (result correct, achieved TFLOPS)."""
    import torch as t

    dev = t.device("cuda", device_index)
    n = 2048
    a = t.randn(n, n, device=dev, dtype=t.float16)
    b = t.randn(n, n, device=dev, dtype=t.float16)
    t.matmul(a, b)  # warm-up
    t.cuda.synchronize(dev)
    reps = 10
    start = time.perf_counter()
    for _ in range(reps):
        c = t.matmul(a, b)
    t.cuda.synchronize(dev)
    elapsed = time.perf_counter() - start
    ref = (a[:64].float() @ b[:, :64].float()).to(t.float16)
    ok = bool(t.allclose(c[:64, :64], ref, atol=1.0, rtol=1e-2))
    tflops = (2 * n**3 * reps) / elapsed / 1e12
    del a, b, c, ref
    t.cuda.empty_cache()
    return ok, round(tflops, 2)


def collect(root: Path, *, gpu_benchmark: bool = True) -> EnvReport:
    problems: list[str] = []
    vm = psutil.virtual_memory()
    gpus: list[GpuInfo] = []
    torch_version = torch_cuda = None
    cuda_available = False
    cudnn: int | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cudnn = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        for i in range(torch.cuda.device_count() if cuda_available else 0):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info = GpuInfo(
                index=i,
                name=props.name,
                capability=f"{props.major}.{props.minor}",
                total_vram_gib=round(total / GIB, 2),
                free_vram_gib=round(free / GIB, 2),
                bf16_supported=bool(torch.cuda.is_bf16_supported()),
            )
            if gpu_benchmark:
                ok, tflops = _matmul_benchmark(i)
                info = info.model_copy(update={"matmul_ok": ok, "matmul_tflops_fp16": tflops})
                if not ok:
                    problems.append(f"GPU {i}: matmul result mismatch")
            gpus.append(info)
        if not cuda_available:
            problems.append("CUDA not available to torch")
    except ImportError:
        problems.append("torch not installed (uv sync --group gpu)")

    return EnvReport(
        created_at=datetime.now(UTC),
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        ram_total_gib=round(vm.total / GIB, 2),
        ram_available_gib=round(vm.available / GIB, 2),
        disk_free_gib=round(shutil.disk_usage(root).free / GIB, 1),
        nvidia_driver=_nvidia_driver(),
        torch=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        cudnn=cudnn,
        gpus=gpus,
        problems=problems,
    )
