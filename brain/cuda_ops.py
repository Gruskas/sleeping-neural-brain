from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import torch

_logger = logging.getLogger(__name__)

# Global lock serializing CUDA Graph capture against background LivingBrainClock GPU work
CUDA_GRAPH_CAPTURE_LOCK = threading.RLock()

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_BUILD_DIR = _CSRC_DIR / "build"

HAS_CUDA_EXT: bool = False
CUDA_EXT_ERROR: str = ""
_EXT: Any = None


def _is_valid_cuda_home(path: Path) -> bool:
    if not path.is_dir():
        return False
    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    has_nvcc = (path / "bin" / nvcc_name).is_file()
    has_hdr = (path / "include" / "cuda_runtime.h").is_file()
    return has_nvcc and has_hdr


def _find_valid_cuda_home() -> Path | None:
    candidates: list[Path] = []
    for env_key in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(env_key)
        if v:
            candidates.append(Path(v))

    which_nvcc = shutil.which("nvcc")
    if which_nvcc:
        candidates.append(Path(which_nvcc).resolve().parent.parent)

    prefix = Path(sys.prefix)
    candidates.append(prefix / "cuda128")
    candidates.extend(sorted(prefix.glob("cuda*"), reverse=True))

    if os.name == "nt":
        for p in sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"), reverse=True):
            candidates.append(Path(p))
    else:
        candidates.append(Path("/usr/local/cuda"))

    for cand in candidates:
        if _is_valid_cuda_home(cand):
            return cand
    return None


def _setup_windows_msvc_env() -> bool:
    if shutil.which("cl.exe"):
        return True

    vcvars_candidates: list[Path] = []
    for pattern in (
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ):
        for match in sorted(glob.glob(pattern), reverse=True):
            p = Path(match)
            if p not in vcvars_candidates:
                vcvars_candidates.append(p)

    sdk_inc_dir = Path(r"C:\Program Files (x86)\Windows Kits\10\include")
    sdk_args: list[str] = []
    if sdk_inc_dir.is_dir():
        sdks = sorted((d.name for d in sdk_inc_dir.iterdir() if d.is_dir()), reverse=True)
        # Prefer Windows SDK 22621 / 22000 first (CUDA 12.8 cudafe++ crashes on 10.0.26100.0 UCRT headers)
        sdk_args.extend([s for s in sdks if not s.startswith("10.0.26100")])
    sdk_args.append("")

    for vcvars in vcvars_candidates:
        vc_tools_dir = vcvars.parent.parent.parent / "Tools" / "MSVC"
        ver_args: list[str] = []
        if vc_tools_dir.is_dir():
            installed_vers = sorted((d.name for d in vc_tools_dir.iterdir() if d.is_dir()), reverse=True)
            # Prefer VS 2022 / CUDA-12.8-compatible toolsets (14.4x / 14.3x) using major.minor
            for v in installed_vers:
                if v.startswith(("14.4", "14.3")):
                    mm = ".".join(v.split(".")[:2])
                    for cand_ver in (mm, v):
                        arg = f"-vcvars_ver={cand_ver}"
                        if arg not in ver_args:
                            ver_args.append(arg)
        ver_args.append("")

        for s_arg in sdk_args:
            for v_arg in ver_args:
                args_str = " ".join(x for x in (s_arg, v_arg) if x)
                inner = f'call "{vcvars}" {args_str} >nul 2>&1 && set' if args_str else f'call "{vcvars}" >nul 2>&1 && set'
                cmd_line = f'cmd.exe /d /c "{inner}"'
                try:
                    proc = subprocess.run(
                        cmd_line,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if proc.returncode != 0:
                        continue
                    for line in proc.stdout.splitlines():
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k and k.upper() != "PROMPT":
                            os.environ[k] = v
                    if shutil.which("cl.exe"):
                        os.environ["DISTUTILS_USE_SDK"] = "1"
                        os.environ["MSSdk"] = "1"
                        return True
                except Exception:
                    continue
    return bool(shutil.which("cl.exe"))


def _load_cuda_extension() -> Any:
    global HAS_CUDA_EXT, CUDA_EXT_ERROR

    if not torch.cuda.is_available():
        CUDA_EXT_ERROR = "CUDA device unavailable"
        return None

    sources = [
        _CSRC_DIR / "bindings.cpp",
        _CSRC_DIR / "fp8_synapse_kernels.cu",
        _CSRC_DIR / "peer_kernels.cu",
        _CSRC_DIR / "peer_cache.cpp",
    ]
    if not all(p.is_file() for p in sources):
        CUDA_EXT_ERROR = f"Missing C++/CUDA sources in {_CSRC_DIR}"
        _logger.warning("Native CUDA extension disabled: %s", CUDA_EXT_ERROR)
        return None

    cuda_home = _find_valid_cuda_home()
    if cuda_home is None:
        CUDA_EXT_ERROR = "nvcc / cuda_runtime.h not found"
        _logger.warning("Native CUDA extension disabled (%s); using PyTorch fallback.", CUDA_EXT_ERROR)
        return None

    if os.name == "nt" and not _setup_windows_msvc_env():
        CUDA_EXT_ERROR = "MSVC cl.exe not found"
        _logger.warning("Native CUDA extension disabled (%s); using PyTorch fallback.", CUDA_EXT_ERROR)
        return None

    try:
        import torch.utils.cpp_extension as cpp_ext

        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["CUDA_PATH"] = str(cuda_home)
        os.environ["PATH"] = str(cuda_home / "bin") + os.pathsep + os.environ.get("PATH", "")
        cpp_ext.CUDA_HOME = str(cuda_home)

        major, minor = torch.cuda.get_device_capability(0)
        arch_str = f"{major}.{minor}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str

        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
        extra_cflags = ["/O2", "/std:c++17", "/EHsc"] if os.name == "nt" else ["-O3", "-std=c++17"]
        extra_cuda_cflags = [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
        ]
        if os.name == "nt":
            extra_cuda_cflags.extend(["-allow-unsupported-compiler", "-Xcompiler", "/EHsc"])

        ext = cpp_ext.load(
            name="_brain_cuda_ext",
            sources=[str(p) for p in sources],
            build_directory=str(_BUILD_DIR),
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=[str(cuda_home / "include")],
            extra_ldflags=[str(cuda_home / "lib" / "x64" / "cudart.lib")] if os.name == "nt" else [],
            verbose=False,
        )
        HAS_CUDA_EXT = True
        CUDA_EXT_ERROR = ""
        _logger.info("Loaded native C++/CUDA extension _brain_cuda_ext (SM %s, CUDA=%s).", arch_str, cuda_home)
        return ext
    except Exception as exc:
        HAS_CUDA_EXT = False
        CUDA_EXT_ERROR = str(exc)
        _logger.warning("Failed to build/load _brain_cuda_ext (%s); falling back to PyTorch.", exc)
        return None


_EXT = _load_cuda_extension()


def quantize_rows_cuda(src: torch.Tensor, mode_id: int, stochastic: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _EXT.quantize_rows_cuda(src, int(mode_id), bool(stochastic), int(seed))


def shy_prune_rows_cuda(
    q_data: torch.Tensor,
    row_scale: torch.Tensor,
    downscale: float,
    eps: float,
    zero_diag: bool,
) -> int:
    return int(_EXT.shy_prune_rows_cuda(q_data, row_scale, float(downscale), float(eps), bool(zero_diag)))


def peer_forward_cuda(
    x_flat: torch.Tensor,
    slot_ids: torch.Tensor,
    weights: torch.Tensor,
    cache_u: torch.Tensor,
    cache_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _EXT.peer_forward_cuda(x_flat, slot_ids, weights, cache_u, cache_v)


def peer_backward_update_cuda(
    grad_out: torch.Tensor,
    x_flat: torch.Tensor,
    slot_ids: torch.Tensor,
    weights: torch.Tensor,
    pre_act: torch.Tensor,
    cache_u: torch.Tensor,
    cache_v: torch.Tensor,
    slot_dirty: torch.Tensor,
    lr: float,
    weight_decay: float,
    do_update: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _EXT.peer_backward_update_cuda(
        grad_out,
        x_flat,
        slot_ids,
        weights,
        pre_act,
        cache_u,
        cache_v,
        slot_dirty,
        float(lr),
        float(weight_decay),
        bool(do_update),
        int(seed),
    )


def peer_cache_ensure_cpp(
    expert_ids: torch.Tensor,
    uniq_ids: torch.Tensor,
    cache_u: torch.Tensor,
    cache_v: torch.Tensor,
    expert_to_slot: torch.Tensor,
    slot_to_expert: torch.Tensor,
    slot_last_step: torch.Tensor,
    slot_dirty: torch.Tensor,
    ram_u_q: torch.Tensor,
    ram_u_scale: torch.Tensor,
    ram_v_q: torch.Tensor,
    ram_v_scale: torch.Tensor,
    stage_u_q: torch.Tensor,
    stage_u_scale: torch.Tensor,
    stage_v_q: torch.Tensor,
    stage_v_scale: torch.Tensor,
    step_counter: int,
    seed: int,
) -> tuple[torch.Tensor, int, int, int]:
    slot_ids, hits, misses, new_step = _EXT.peer_cache_ensure_cpp(
        expert_ids,
        uniq_ids,
        cache_u,
        cache_v,
        expert_to_slot,
        slot_to_expert,
        slot_last_step,
        slot_dirty,
        ram_u_q,
        ram_u_scale,
        ram_v_q,
        ram_v_scale,
        stage_u_q,
        stage_u_scale,
        stage_v_q,
        stage_v_scale,
        int(step_counter),
        int(seed),
    )
    return slot_ids, int(hits), int(misses), int(new_step)


def peer_flush_dirty_cpp(
    cache_u: torch.Tensor,
    cache_v: torch.Tensor,
    slot_to_expert: torch.Tensor,
    slot_dirty: torch.Tensor,
    ram_u_q: torch.Tensor,
    ram_u_scale: torch.Tensor,
    ram_v_q: torch.Tensor,
    ram_v_scale: torch.Tensor,
    stage_u_q: torch.Tensor,
    stage_u_scale: torch.Tensor,
    stage_v_q: torch.Tensor,
    stage_v_scale: torch.Tensor,
    seed: int,
) -> int:
    return int(
        _EXT.peer_flush_dirty_cpp(
            cache_u,
            cache_v,
            slot_to_expert,
            slot_dirty,
            ram_u_q,
            ram_u_scale,
            ram_v_q,
            ram_v_scale,
            stage_u_q,
            stage_u_scale,
            stage_v_q,
            stage_v_scale,
            int(seed),
        )
    )
