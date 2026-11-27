from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .cuda_ops import HAS_CUDA_EXT, quantize_rows_cuda, shy_prune_rows_cuda

FP8_E4M3_MAX: float = 448.0
INT8_MAX: float = 127.0
_MIN_SCALE: float = 1e-8
_MODE_CACHE: dict[tuple[str, str], str] = {}
_CUDA_SEED_COUNTER: int = 1337


def _next_cuda_seed() -> int:
    global _CUDA_SEED_COUNTER
    _CUDA_SEED_COUNTER += 1
    return (_CUDA_SEED_COUNTER * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF


def resolve_bio_synapse_mode(requested: str, device: torch.device) -> str:
    """
    Resolve requested biological synapse storage mode (`fp8` | `int8` | `bf16` | `fp32`)
    with automatic fallback to `int8` or `bf16` if the device/PyTorch build does not
    support `torch.float8_e4m3fn`.
    """
    req = (requested or "fp8").strip().lower()
    if req in ("bf16", "bfloat16"):
        return "bf16" if (device.type == "cpu" or torch.cuda.is_bf16_supported()) else "fp32"
    if req in ("fp32", "float32"):
        return "fp32"
    if req == "int8":
        return "int8"

    key = (req, device.type)
    cached = _MODE_CACHE.get(key)
    if cached is not None:
        return cached

    # Probe native torch.float8_e4m3fn support on target device
    if hasattr(torch, "float8_e4m3fn"):
        try:
            probe = torch.tensor([1.0, -2.5], device=device, dtype=torch.float32)
            q = probe.to(torch.float8_e4m3fn)
            work_dt = torch.bfloat16 if (device.type == "cpu" or torch.cuda.is_bf16_supported()) else torch.float32
            back = q.to(dtype=work_dt)
            if back.shape == probe.shape and bool(torch.isfinite(back.float()).all().item()):
                _MODE_CACHE[key] = "fp8"
                return "fp8"
        except Exception:
            pass

    # Fallback: 1-byte per-row int8 quantization (still 2x VRAM savings vs bf16)
    try:
        probe_i8 = torch.tensor([1, -2], device=device, dtype=torch.int8)
        _ = probe_i8.to(dtype=torch.float32)
        _MODE_CACHE[key] = "int8"
        return "int8"
    except Exception:
        fallback = "bf16" if (device.type == "cpu" or torch.cuda.is_bf16_supported()) else "fp32"
        _MODE_CACHE[key] = fallback
        return fallback


def _work_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


@torch.no_grad()
def _quantize_rows(
    src_f32: torch.Tensor,
    mode: str,
    stochastic: bool = False,
    _force_pytorch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Per-row quantization into `float8_e4m3fn` (range [-448, 448]), `int8` ([-127, 127]),
    or passthrough `bf16`/`fp32`.
    When `stochastic=True`, applies zero-mean 1-ULP unbiased stochastic rounding on non-zero
    entries so small Oja-Hebb / Predictive Coding steps never vanish under 3-bit E4M3 mantissa!
    Exact zeros stay exact zeros (preserving sparsity & zero-diagonal invariants).
    """
    if src_f32.ndim != 2:
        raise ValueError(f"QuantizedSynapseMatrix expects 2D tensor, got shape {tuple(src_f32.shape)}")
    rows, cols = src_f32.shape
    dev = src_f32.device

    if mode == "bf16":
        return src_f32.to(dtype=torch.bfloat16).contiguous(), None
    if mode == "fp32":
        return src_f32.to(dtype=torch.float32).contiguous(), None

    if rows == 0 or cols == 0:
        dt = torch.float8_e4m3fn if mode == "fp8" else torch.int8
        return (
            torch.empty((rows, cols), device=dev, dtype=dt),
            torch.ones((rows,), device=dev, dtype=torch.float32),
        )

    if (not _force_pytorch) and HAS_CUDA_EXT and src_f32.is_cuda and mode in ("fp8", "int8"):
        mode_id = 0 if mode == "fp8" else 1
        seed = _next_cuda_seed() if stochastic else 0
        return quantize_rows_cuda(src_f32, mode_id, bool(stochastic), seed)

    f = torch.nan_to_num(src_f32.float(), nan=0.0, posinf=0.0, neginf=0.0)
    amax = f.abs().amax(dim=-1)  # [rows]

    if mode == "fp8":
        scale = (amax / FP8_E4M3_MAX).clamp(min=_MIN_SCALE)  # [rows] fp32
        scaled = (f / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        if stochastic:
            nonzero = f != 0
            # Exact E4M3 ULP: 3 mantissa bits -> ulp = 2^(floor(log2(|x|)) - 3), subnormal min = 2^-9
            abs_s = scaled.abs().clamp(min=2.0 ** -6)
            exp_floor = torch.floor(torch.log2(abs_s))
            ulp = torch.ldexp(torch.ones_like(scaled), (exp_floor - 3.0).to(torch.int32)).clamp(min=2.0 ** -9)
            noise = (torch.rand_like(scaled) - 0.5) * ulp
            scaled = torch.where(nonzero, (scaled + noise).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX), scaled)
        q = scaled.to(torch.float8_e4m3fn).contiguous()
        return q, scale.contiguous()

    # mode == "int8"
    scale = (amax / INT8_MAX).clamp(min=_MIN_SCALE)
    scaled = (f / scale.unsqueeze(-1)).clamp(-INT8_MAX, INT8_MAX)
    if stochastic:
        nonzero = f != 0
        noise = torch.rand_like(scaled) - 0.5
        scaled = torch.where(nonzero, (scaled + noise).clamp(-INT8_MAX, INT8_MAX), scaled)
    q = scaled.round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8).contiguous()
    return q, scale.contiguous()


def tensor_or_qmat_bytes(obj: Any) -> int:
    """Exact byte size for either a torch.Tensor or QuantizedSynapseMatrix (1B/param + row_scale for FP8/INT8)."""
    if isinstance(obj, QuantizedSynapseMatrix):
        return obj.nbytes()
    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    return 0


class QuantizedSynapseMatrix:
    """
    Lightweight per-row quantized 2D biological synapse / engram matrix optimized for
    NVIDIA Blackwell (RTX 5080, SM 12.0, CUDA 12.8):
      - Stores weights in native `torch.float8_e4m3fn` (1 byte/param) + per-row `row_scale` (fp32).
      - Automatic fallback to `int8` (1 byte/param) or `bfloat16` on unsupported hardware.
      - Fast inference via native Blackwell `torch._scaled_mm` FP8 Tensor Cores (when aligned)
        or blockwise `bfloat16` dequantization (never allocates a full `float32` copy!).
      - Safe local biological updates (Oja-Hebb, Predictive Coding, Hebbian, SHY pruning) with
        E4M3 range clamping (`[-448, 448]`) and unbiased stochastic rounding to preserve small `lr` steps.
    """

    __slots__ = ("data", "row_scale", "mode", "_one_scalar")

    def __init__(
        self,
        data: torch.Tensor,
        row_scale: torch.Tensor | None,
        mode: str = "fp8",
    ) -> None:
        self.data = data
        self.row_scale = row_scale
        self.mode = mode
        self._one_scalar: torch.Tensor | None = None

    # ── Constructors & Serialization ─────────────────────────────────────────
    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        mode: str = "fp8",
        device: torch.device | None = None,
        stochastic: bool = False,
        _force_pytorch: bool = False,
    ) -> "QuantizedSynapseMatrix":
        dev = device or tensor.device
        resolved = resolve_bio_synapse_mode(mode, dev)
        if (
            (not _force_pytorch)
            and HAS_CUDA_EXT
            and dev.type == "cuda"
            and resolved in ("fp8", "int8")
            and tensor.dtype in (torch.float32, torch.bfloat16, torch.float16)
        ):
            src = tensor.to(device=dev) if tensor.device != dev else tensor
        else:
            src = tensor.to(device=dev, dtype=torch.float32) if tensor.device != dev or tensor.dtype != torch.float32 else tensor
        q, s = _quantize_rows(src, resolved, stochastic=stochastic, _force_pytorch=_force_pytorch)
        return cls(q, s, mode=resolved)

    @classmethod
    def empty(cls, rows: int, cols: int, device: torch.device, mode: str = "fp8") -> "QuantizedSynapseMatrix":
        resolved = resolve_bio_synapse_mode(mode, device)
        if resolved == "fp8":
            return cls(
                torch.empty((rows, cols), device=device, dtype=torch.float8_e4m3fn),
                torch.ones((rows,), device=device, dtype=torch.float32),
                mode="fp8",
            )
        if resolved == "int8":
            return cls(
                torch.empty((rows, cols), device=device, dtype=torch.int8),
                torch.ones((rows,), device=device, dtype=torch.float32),
                mode="int8",
            )
        dt = torch.bfloat16 if resolved == "bf16" else torch.float32
        return cls(torch.empty((rows, cols), device=device, dtype=dt), None, mode=resolved)

    @classmethod
    def zeros(cls, rows: int, cols: int, device: torch.device, mode: str = "fp8") -> "QuantizedSynapseMatrix":
        resolved = resolve_bio_synapse_mode(mode, device)
        if resolved == "fp8":
            return cls(
                torch.zeros((rows, cols), device=device, dtype=torch.float8_e4m3fn),
                torch.full((rows,), _MIN_SCALE, device=device, dtype=torch.float32),
                mode="fp8",
            )
        if resolved == "int8":
            return cls(
                torch.zeros((rows, cols), device=device, dtype=torch.int8),
                torch.full((rows,), _MIN_SCALE, device=device, dtype=torch.float32),
                mode="int8",
            )
        dt = torch.bfloat16 if resolved == "bf16" else torch.float32
        return cls(torch.zeros((rows, cols), device=device, dtype=dt), None, mode=resolved)

    def export_state(self) -> dict[str, Any]:
        return {
            "__qsyn__": True,
            "mode": self.mode,
            "data": self.data.detach().cpu(),
            "row_scale": self.row_scale.detach().cpu().float() if self.row_scale is not None else None,
        }

    @classmethod
    @torch.no_grad()
    def from_state(
        cls,
        obj: Any,
        device: torch.device,
        mode: str = "fp8",
    ) -> "QuantizedSynapseMatrix":
        """
        Load either a quantized FP8/INT8 state dict OR a legacy bf16/fp32/fp16 torch.Tensor,
        automatically converting legacy checkpoints into native FP8 per-row without RuntimeError.
        """
        resolved = resolve_bio_synapse_mode(mode, device)
        if isinstance(obj, QuantizedSynapseMatrix):
            return obj.to(device=device, mode=resolved)
        if isinstance(obj, dict) and obj.get("__qsyn__"):
            saved_mode = str(obj.get("mode", "fp8"))
            saved_data = obj["data"]
            saved_scale = obj.get("row_scale")
            if saved_mode == resolved:
                q = saved_data.to(device=device).contiguous()
                s = saved_scale.to(device=device, dtype=torch.float32).contiguous() if saved_scale is not None else None
                return cls(q, s, mode=resolved)
            # Re-quantize cleanly across different storage modes
            tmp = cls(
                saved_data.to(device=device),
                saved_scale.to(device=device, dtype=torch.float32) if saved_scale is not None else None,
                mode=saved_mode,
            )
            return cls.from_tensor(tmp.float(), mode=resolved, device=device)
        if isinstance(obj, torch.Tensor):
            return cls.from_tensor(obj, mode=resolved, device=device)
        raise TypeError(f"Unsupported QuantizedSynapseMatrix state type: {type(obj).__name__}")

    # ── Tensor-like Properties & Memory Accounting ───────────────────────────
    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def requires_grad(self) -> bool:
        return False

    def numel(self) -> int:
        return int(self.data.numel())

    def element_size(self) -> int:
        return int(self.data.element_size())

    def nbytes(self) -> int:
        total = int(self.data.numel() * self.data.element_size())
        if self.row_scale is not None:
            total += int(self.row_scale.numel() * self.row_scale.element_size())
        return total

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def clone(self) -> "QuantizedSynapseMatrix":
        return QuantizedSynapseMatrix(
            self.data.clone(),
            self.row_scale.clone() if self.row_scale is not None else None,
            mode=self.mode,
        )

    def detach(self) -> "QuantizedSynapseMatrix":
        return QuantizedSynapseMatrix(
            self.data.detach(),
            self.row_scale.detach() if self.row_scale is not None else None,
            mode=self.mode,
        )

    def contiguous(self) -> "QuantizedSynapseMatrix":
        self.data = self.data.contiguous()
        if self.row_scale is not None:
            self.row_scale = self.row_scale.contiguous()
        return self

    @torch.no_grad()
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        mode: str | None = None,
    ) -> "QuantizedSynapseMatrix | torch.Tensor":
        target_dev = torch.device(device) if device is not None else self.device
        # If caller explicitly requests a dense floating dtype (e.g. .to(torch.float32) or .to(torch.bfloat16)),
        # return a dequantized tensor for drop-in compatibility.
        if dtype is not None and mode is None:
            if dtype in (torch.float32, torch.bfloat16, torch.float16):
                return self.dequant_all(out_dtype=dtype).to(device=target_dev)
            if dtype == torch.float8_e4m3fn:
                mode = "fp8"
            elif dtype == torch.int8:
                mode = "int8"

        target_mode = resolve_bio_synapse_mode(mode or self.mode, target_dev)
        if target_mode == self.mode:
            return QuantizedSynapseMatrix(
                self.data.to(device=target_dev).contiguous(),
                self.row_scale.to(device=target_dev, dtype=torch.float32).contiguous() if self.row_scale is not None else None,
                mode=target_mode,
            )
        return QuantizedSynapseMatrix.from_tensor(self.float(), mode=target_mode, device=target_dev)

    @torch.no_grad()
    def copy_(self, src: "QuantizedSynapseMatrix | torch.Tensor", stochastic: bool = False) -> "QuantizedSynapseMatrix":
        if isinstance(src, QuantizedSynapseMatrix):
            if src.mode == self.mode and src.shape == self.shape:
                self.data = src.data.to(device=self.device).contiguous()
                self.row_scale = (
                    src.row_scale.to(device=self.device, dtype=torch.float32).contiguous()
                    if src.row_scale is not None
                    else None
                )
                return self
            q, s = _quantize_rows(src.float().to(device=self.device), self.mode, stochastic=stochastic)
            self.data, self.row_scale = q, s
            return self
        q, s = _quantize_rows(src.to(device=self.device, dtype=torch.float32), self.mode, stochastic=stochastic)
        self.data, self.row_scale = q, s
        return self

    # ── Dequantization Helpers (Row/Slice Scoped, Zero Unnecessary Full Copies) ──
    @torch.no_grad()
    def dequant_rows(self, indices: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        idx = indices.to(device=self.device, dtype=torch.long)
        sub = self.data.index_select(0, idx)
        if self.row_scale is None:
            return sub.to(dtype=out_dtype)
        sc = self.row_scale.index_select(0, idx).unsqueeze(-1)
        return (sub.float() * sc).to(dtype=out_dtype)

    @torch.no_grad()
    def dequant_row(self, idx: int, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        row = self.data[idx]
        if self.row_scale is None:
            return row.to(dtype=out_dtype)
        return (row.float() * self.row_scale[idx]).to(dtype=out_dtype)

    @torch.no_grad()
    def dequant_all(self, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.row_scale is None:
            return self.data.to(dtype=out_dtype)
        if self.data.shape[0] == 0:
            return torch.empty(self.shape, device=self.device, dtype=out_dtype)
        return (self.data.float() * self.row_scale.unsqueeze(-1)).to(dtype=out_dtype)

    def float(self) -> torch.Tensor:
        return self.dequant_all(out_dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return self.dequant_all(out_dtype=torch.bfloat16)

    @torch.no_grad()
    def diagonal(self) -> torch.Tensor:
        n = min(self.shape[0], self.shape[1])
        if n == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        d = self.data.diagonal().float()
        if self.row_scale is not None:
            d = d * self.row_scale[:n]
        return d

    @torch.no_grad()
    def abs(self) -> torch.Tensor:
        return self.dequant_all(out_dtype=torch.float32).abs()

    @torch.no_grad()
    def fill_diagonal_(self, val: float = 0.0) -> "QuantizedSynapseMatrix":
        if val == 0.0:
            if self.shape[0] > 0 and self.shape[1] > 0:
                if self.mode == "fp8":
                    # Cast-safe zeroing on diagonal without full dequant
                    n = min(self.shape[0], self.shape[1])
                    idx = torch.arange(n, device=self.device)
                    self.data[idx, idx] = torch.zeros((), device=self.device, dtype=torch.float8_e4m3fn)
                else:
                    self.data.fill_diagonal_(0)
            return self
        f = self.float()
        f.fill_diagonal_(float(val))
        return self.copy_(f)

    @torch.no_grad()
    def count_nonzero(self) -> int:
        if self.data.numel() == 0:
            return 0
        if self.mode == "fp8":
            # In E4M3, +0.0 is 0x00 and -0.0 is 0x80; check via raw uint8 view (zero cast!)
            raw = self.data.view(torch.uint8)
            return int(((raw & 0x7F) != 0).sum().item())
        return int((self.data != 0).sum().item())

    @torch.no_grad()
    def is_all_finite(self) -> bool:
        if self.data.numel() == 0:
            return True
        if self.mode == "fp8":
            # In float8_e4m3fn, NaN is 0x7F or 0xFF (no Inf exists)
            raw = self.data.view(torch.uint8)
            data_ok = not bool(((raw & 0x7F) == 0x7F).any().item())
        elif self.mode == "int8":
            data_ok = True
        else:
            data_ok = bool(torch.isfinite(self.data).all().item())
        if not data_ok:
            return False
        if self.row_scale is not None:
            return bool(torch.isfinite(self.row_scale).all().item())
        return True

    # ── Fast Inference: Native Blackwell FP8 _scaled_mm & Blockwise BF16 ─────
    def _get_one_scalar(self) -> torch.Tensor:
        if self._one_scalar is None or self._one_scalar.device != self.device:
            self._one_scalar = torch.ones((), device=self.device, dtype=torch.float32)
        return self._one_scalar

    @torch.no_grad()
    def matmul_right_t(self, x: torch.Tensor, block_rows: int = 2048) -> torch.Tensor:
        """
        Compute `X @ W.T` -> `[B, M]` float32 WITHOUT allocating a full float32 copy of W:
          - On CUDA with `mode == "fp8"` and `M % 16 == 0, N % 16 == 0`: executes natively on
            Blackwell FP8 Tensor Cores via `torch._scaled_mm` + post per-row scaling!
          - Otherwise: blockwise dequantization into `bfloat16` (`block_rows` at a time).
        """
        m, n = self.shape
        x_2d = x.unsqueeze(0) if x.ndim == 1 else x
        b = x_2d.shape[0]
        if m == 0 or n == 0 or b == 0:
            out = torch.zeros((b, m), device=self.device, dtype=torch.float32)
            return out.squeeze(0) if x.ndim == 1 else out

        # Fast path 1: Native Blackwell FP8 Tensor Core GEMM (zero dequant allocation!)
        if (
            self.mode == "fp8"
            and self.device.type == "cuda"
            and (m % 16 == 0)
            and (n % 16 == 0)
            and hasattr(torch, "_scaled_mm")
            and self.row_scale is not None
        ):
            x_f = x_2d.to(device=self.device, dtype=torch.float32)
            sa = (x_f.abs().amax() / FP8_E4M3_MAX).clamp(min=_MIN_SCALE)
            qx = (x_f / sa).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
            out_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            raw = torch._scaled_mm(qx, self.data.t(), scale_a=sa, scale_b=self._get_one_scalar(), out_dtype=out_dt)
            out = raw.float() * self.row_scale.unsqueeze(0)
            return out.squeeze(0) if x.ndim == 1 else out

        # Fast path 2: Blockwise bfloat16 matmul (zero full fp32 copy!)
        w_dt = _work_dtype(self.device)
        x_w = x_2d.to(device=self.device, dtype=w_dt)
        if self.row_scale is None:
            w_t = self.data.to(dtype=w_dt).t() if self.data.dtype != w_dt else self.data.t()
            out = (x_w @ w_t).float()
            return out.squeeze(0) if x.ndim == 1 else out

        out = torch.empty((b, m), device=self.device, dtype=torch.float32)
        for r0 in range(0, m, block_rows):
            r1 = min(m, r0 + block_rows)
            w_blk = self.data[r0:r1].to(dtype=w_dt)
            out[:, r0:r1] = (x_w @ w_blk.t()).float() * self.row_scale[r0:r1].unsqueeze(0)
        return out.squeeze(0) if x.ndim == 1 else out

    @torch.no_grad()
    def matvec(self, x: torch.Tensor, block_rows: int = 2048) -> torch.Tensor:
        """Compute `W @ x` -> `[M]` float32 via native FP8 Tensor Cores or blockwise BF16."""
        return self.matmul_right_t(x.reshape(-1), block_rows=block_rows)

    @torch.no_grad()
    def matmul_left(self, x: torch.Tensor, block_rows: int = 2048) -> torch.Tensor:
        """
        Compute `X @ W` -> `[B, N]` float32 (or `[N]` if 1D) without full dequant:
        Folds per-row `row_scale` into the small `[B, M]` input `X` before blockwise BF16 matmul!
        """
        m, n = self.shape
        x_2d = x.unsqueeze(0) if x.ndim == 1 else x
        b = x_2d.shape[0]
        if m == 0 or n == 0 or b == 0:
            out = torch.zeros((b, n), device=self.device, dtype=torch.float32)
            return out.squeeze(0) if x.ndim == 1 else out

        w_dt = _work_dtype(self.device)
        x_f = x_2d.to(device=self.device, dtype=torch.float32)
        if self.row_scale is not None:
            x_f = x_f * self.row_scale.unsqueeze(0)
        x_w = x_f.to(dtype=w_dt)

        if m <= block_rows:
            w_blk = self.data if self.data.dtype == w_dt else self.data.to(dtype=w_dt)
            out = (x_w @ w_blk).float()
        else:
            out = torch.zeros((b, n), device=self.device, dtype=torch.float32)
            for r0 in range(0, m, block_rows):
                r1 = min(m, r0 + block_rows)
                w_blk = self.data[r0:r1].to(dtype=w_dt)
                out.add_((x_w[:, r0:r1] @ w_blk).float())
        return out.squeeze(0) if x.ndim == 1 else out

    # ── Safe Biological Local Plasticity Updates (Stochastic Rounding) ───────
    @torch.no_grad()
    def set_rows(self, indices: torch.Tensor, values_f32: torch.Tensor, stochastic: bool = False) -> None:
        if indices.numel() == 0:
            return
        idx = indices.to(device=self.device, dtype=torch.long)
        vals = values_f32.to(device=self.device, dtype=torch.float32)
        if vals.ndim == 1:
            vals = vals.unsqueeze(0)
        q, s = _quantize_rows(vals, self.mode, stochastic=stochastic)
        self.data[idx] = q
        if self.row_scale is not None and s is not None:
            self.row_scale[idx] = s

    @torch.no_grad()
    def oja_hebb_update(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lr: float,
        decay: float = 0.04,
        max_row_norm: float = 1.2,
        zero_diag: bool = False,
    ) -> None:
        """
        Sparse Oja-Hebb local rule updating ONLY active postsynaptic rows `y > 0` in fp32,
        then re-quantizing ONLY those rows back to FP8 with unbiased stochastic rounding!
        """
        active = torch.nonzero(y > 0, as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            return
        y_a = y[active].float().unsqueeze(1)  # [k, 1]
        x_f = x.to(device=self.device, dtype=torch.float32)
        w_sub = self.dequant_rows(active, out_dtype=torch.float32)  # [k, N]
        dw = float(lr) * y_a * (x_f.unsqueeze(0) - float(decay) * y_a * w_sub)
        w_new = w_sub + dw
        if zero_diag and self.shape[0] == self.shape[1]:
            w_new[torch.arange(active.numel(), device=self.device), active] = 0.0
        norms = w_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        w_new = w_new * torch.clamp(float(max_row_norm) / norms, max=1.0)
        self.set_rows(active, w_new, stochastic=True)

    @torch.no_grad()
    def local_predictive_update(
        self,
        err: torch.Tensor,
        src_sdr: torch.Tensor,
        lr: float,
        max_col_norm: float = 1.2,
    ) -> None:
        """Local Predictive Coding update on active presynaptic columns, re-quantized with stochastic rounding."""
        active = torch.nonzero(src_sdr > 0, as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            return
        # M1 generative matrices are modest (<= few thousand rows); update active cols cleanly
        g_f = self.float()
        g_sub = g_f.index_select(1, active)
        dg = float(lr) * torch.outer(err.to(device=self.device, dtype=torch.float32), src_sdr[active].float())
        g_new = g_sub + dg
        norms = g_new.norm(dim=0, keepdim=True).clamp(min=1e-6)
        g_new = g_new * torch.clamp(float(max_col_norm) / norms, max=1.0)
        g_f[:, active] = g_new
        self.copy_(g_f, stochastic=True)

    @torch.no_grad()
    def hebbian_update_from_sdr(
        self,
        z: torch.Tensor,
        lr: float,
        decay: float,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
        zero_diag: bool = True,
        block_rows: int = 2048,
    ) -> None:
        """
        Blockwise Hebbian update `W += lr * ((z^T @ z)/B - decay * W)` with stochastic FP8 rounding.
        Peak temporary memory for 12288x12288 is ~100 MB instead of ~1.2 GB!
        """
        m, n = self.shape
        b = max(1, int(z.shape[0]))
        z_f = z.to(device=self.device, dtype=torch.float32)
        lr_f = float(lr)
        dec_f = float(decay)
        for r0 in range(0, m, block_rows):
            r1 = min(m, r0 + block_rows)
            idx = torch.arange(r0, r1, device=self.device)
            w_blk = self.dequant_rows(idx, out_dtype=torch.float32)
            coact_blk = (z_f[:, r0:r1].t() @ z_f) / float(b)
            w_blk.add_(lr_f * (coact_blk - dec_f * w_blk))
            if zero_diag and m == n:
                local_i = torch.arange(r1 - r0, device=self.device)
                w_blk[local_i, local_i + r0] = 0.0
            w_blk.clamp_(clamp_min, clamp_max)
            self.set_rows(idx, w_blk, stochastic=True)

    @torch.no_grad()
    def add_scaled_and_clamp(
        self,
        delta: "QuantizedSynapseMatrix | torch.Tensor",
        alpha: float,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
        zero_diag: bool = False,
        block_rows: int = 2048,
    ) -> None:
        m, n = self.shape
        for r0 in range(0, m, block_rows):
            r1 = min(m, r0 + block_rows)
            idx = torch.arange(r0, r1, device=self.device)
            w_blk = self.dequant_rows(idx, out_dtype=torch.float32)
            d_blk = (
                delta.dequant_rows(idx, out_dtype=torch.float32)
                if isinstance(delta, QuantizedSynapseMatrix)
                else delta[r0:r1].to(device=self.device, dtype=torch.float32)
            )
            w_blk.add_(d_blk, alpha=float(alpha))
            if zero_diag and m == n:
                local_i = torch.arange(r1 - r0, device=self.device)
                w_blk[local_i, local_i + r0] = 0.0
            w_blk.clamp_(clamp_min, clamp_max)
            self.set_rows(idx, w_blk, stochastic=True)

    @torch.no_grad()
    def decay_and_add_submatrix(
        self,
        decay_factor: float,
        active_idx: torch.Tensor,
        delta_sub: torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        """Fast eligibility trace update: global multiplicative decay on `row_scale` + sparse active block add."""
        if self.row_scale is not None:
            self.row_scale.mul_(float(decay_factor)).clamp_(min=_MIN_SCALE)
        else:
            self.data.mul_(float(decay_factor))
        if active_idx.numel() == 0:
            return
        idx = active_idx.to(device=self.device, dtype=torch.long)
        rows_f = self.dequant_rows(idx, out_dtype=torch.float32)  # [k, N]
        rows_f[:, idx] += float(alpha) * delta_sub.to(device=self.device, dtype=torch.float32)
        self.set_rows(idx, rows_f, stochastic=True)

    @torch.no_grad()
    def apply_shy_prune(
        self,
        downscale: float = 0.94,
        eps: float = 0.008,
        zero_diag: bool = True,
        block_rows: int = 2048,
        _force_pytorch: bool = False,
    ) -> dict[str, int]:
        """
        Synaptic Homeostasis Hypothesis (SHY) in FP8:
          - Exact multiplicative downscaling via `row_scale *= downscale` (zero rounding loss!).
          - Blockwise pruning of weak synapses `|w| < eps` and zero-diagonal enforcement.
        """
        if zero_diag and self.shape[0] == self.shape[1]:
            self.fill_diagonal_(0.0)
        before = self.count_nonzero()
        if (
            (not _force_pytorch)
            and HAS_CUDA_EXT
            and self.device.type == "cuda"
            and self.mode in ("fp8", "int8")
            and self.row_scale is not None
        ):
            m, n = self.shape
            self.data = self.data.contiguous()
            self.row_scale = self.row_scale.contiguous()
            pruned = shy_prune_rows_cuda(
                self.data,
                self.row_scale,
                float(downscale),
                float(eps),
                bool(zero_diag and m == n),
            )
            after = max(0, before - pruned)
            return {
                "before_synapses": before,
                "pruned_synapses": pruned,
                "active_synapses": after,
            }

        if self.row_scale is not None:
            self.row_scale.mul_(float(downscale)).clamp_(min=_MIN_SCALE)
        else:
            self.data.mul_(float(downscale))

        m, n = self.shape
        pruned = 0
        zero_scalar = torch.zeros((), device=self.device, dtype=self.data.dtype)
        for r0 in range(0, m, block_rows):
            r1 = min(m, r0 + block_rows)
            idx = torch.arange(r0, r1, device=self.device)
            w_blk = self.dequant_rows(idx, out_dtype=torch.float32)
            mask = (w_blk.abs() < float(eps)) & (w_blk != 0)
            c = int(mask.sum().item())
            if c > 0:
                self.data[r0:r1] = torch.where(mask, zero_scalar, self.data[r0:r1])
                pruned += c

        if zero_diag and m == n:
            self.fill_diagonal_(0.0)
        after = self.count_nonzero()
        return {
            "before_synapses": before,
            "pruned_synapses": pruned,
            "active_synapses": after,
        }

    # ── Structural Ops (Growth, Compaction, 2D Graph Projection) ─────────────
    @torch.no_grad()
    def select_rows(self, indices: torch.Tensor) -> "QuantizedSynapseMatrix":
        idx = indices.to(device=self.device, dtype=torch.long)
        return QuantizedSynapseMatrix(
            self.data.index_select(0, idx).contiguous(),
            self.row_scale.index_select(0, idx).contiguous() if self.row_scale is not None else None,
            mode=self.mode,
        )

    @torch.no_grad()
    def select_cols(self, indices: torch.Tensor) -> "QuantizedSynapseMatrix":
        idx = indices.to(device=self.device, dtype=torch.long)
        return QuantizedSynapseMatrix(
            self.data.index_select(1, idx).contiguous(),
            self.row_scale.clone() if self.row_scale is not None else None,
            mode=self.mode,
        )

    @torch.no_grad()
    def select_square(self, indices: torch.Tensor, zero_diag: bool = True) -> "QuantizedSynapseMatrix":
        idx = indices.to(device=self.device, dtype=torch.long)
        out = QuantizedSynapseMatrix(
            self.data.index_select(0, idx).index_select(1, idx).contiguous(),
            self.row_scale.index_select(0, idx).contiguous() if self.row_scale is not None else None,
            mode=self.mode,
        )
        if zero_diag:
            out.fill_diagonal_(0.0)
        return out

    @torch.no_grad()
    def index_select(self, dim: int, index: torch.Tensor) -> "QuantizedSynapseMatrix":
        if dim == 0:
            return self.select_rows(index)
        if dim == 1:
            return self.select_cols(index)
        raise ValueError(f"Invalid dim {dim} for 2D QuantizedSynapseMatrix")

    @torch.no_grad()
    def cat_rows(self, other: "QuantizedSynapseMatrix | torch.Tensor") -> "QuantizedSynapseMatrix":
        rhs = other if (isinstance(other, QuantizedSynapseMatrix) and other.mode == self.mode) else (
            QuantizedSynapseMatrix.from_tensor(
                other.float() if isinstance(other, QuantizedSynapseMatrix) else other,
                mode=self.mode,
                device=self.device,
            )
        )
        if self.shape[0] == 0:
            return rhs.clone()
        if rhs.shape[0] == 0:
            return self.clone()
        new_data = torch.cat([self.data, rhs.data.to(device=self.device)], dim=0).contiguous()
        new_scale = (
            torch.cat([self.row_scale, rhs.row_scale.to(device=self.device)], dim=0).contiguous()
            if (self.row_scale is not None and rhs.row_scale is not None)
            else None
        )
        return QuantizedSynapseMatrix(new_data, new_scale, mode=self.mode)

    @torch.no_grad()
    def cat_cols(self, other_f32: torch.Tensor) -> "QuantizedSynapseMatrix":
        rhs = other_f32.to(device=self.device, dtype=torch.float32)
        if self.row_scale is None:
            new_data = torch.cat([self.data, rhs.to(dtype=self.data.dtype)], dim=1).contiguous()
            return QuantizedSynapseMatrix(new_data, None, mode=self.mode)
        # Re-quantize concatenated rows so per-row scale covers newly appended columns cleanly
        merged = torch.cat([self.float(), rhs], dim=1)
        return QuantizedSynapseMatrix.from_tensor(merged, mode=self.mode, device=self.device)

    @torch.no_grad()
    def grow_square(
        self,
        bottom_left: torch.Tensor,
        top_right: torch.Tensor,
        bottom_right: torch.Tensor,
        zero_diag: bool = True,
    ) -> "QuantizedSynapseMatrix":
        top = self.cat_cols(top_right)
        bottom = torch.cat(
            [
                bottom_left.to(device=self.device, dtype=torch.float32),
                bottom_right.to(device=self.device, dtype=torch.float32),
            ],
            dim=1,
        )
        out = top.cat_rows(bottom)
        if zero_diag:
            out.fill_diagonal_(0.0)
        return out

    @torch.no_grad()
    def row_col_abs_sum(self, block_rows: int = 2048) -> torch.Tensor:
        """Compute `W.abs().sum(dim=1) + W.abs().sum(dim=0)` blockwise without full dequant."""
        m, n = self.shape
        row_sum = torch.zeros((m,), device=self.device, dtype=torch.float32)
        col_sum = torch.zeros((n,), device=self.device, dtype=torch.float32)
        for r0 in range(0, m, block_rows):
            r1 = min(m, r0 + block_rows)
            idx = torch.arange(r0, r1, device=self.device)
            blk_abs = self.dequant_rows(idx, out_dtype=torch.float32).abs()
            row_sum[r0:r1] = blk_abs.sum(dim=1)
            col_sum.add_(blk_abs.sum(dim=0))
        return row_sum + col_sum if m == n else row_sum

    @torch.no_grad()
    def select_submatrix(self, row_idx: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
        rows_f = self.dequant_rows(row_idx, out_dtype=torch.float32)
        return rows_f.index_select(1, col_idx.to(device=self.device, dtype=torch.long))

    # ── Indexing & __torch_function__ Drop-In Interop ────────────────────────
    @torch.no_grad()
    def __getitem__(self, key: Any) -> torch.Tensor:
        if isinstance(key, int):
            return self.dequant_row(key, out_dtype=torch.float32)
        if isinstance(key, tuple) and len(key) == 2:
            r_k, c_k = key
            if isinstance(r_k, int):
                return self.dequant_row(r_k, out_dtype=torch.float32)[c_k]
            sub = self.data[r_k, c_k]
            if self.row_scale is None:
                return sub.float()
            sc = self.row_scale[r_k]
            if sc.ndim < sub.ndim:
                sc = sc.unsqueeze(-1)
            return sub.float() * sc
        if isinstance(key, torch.Tensor):
            if key.dtype == torch.bool:
                idx = torch.nonzero(key, as_tuple=False).squeeze(-1)
                return self.dequant_rows(idx, out_dtype=torch.float32)
            return self.dequant_rows(key, out_dtype=torch.float32)
        sub = self.data[key]
        if self.row_scale is None:
            return sub.float()
        sc = self.row_scale[key]
        return sub.float() * sc.unsqueeze(-1)

    @torch.no_grad()
    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            val_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device, dtype=torch.float32)
            self.set_rows(torch.tensor([key], device=self.device, dtype=torch.long), val_t.reshape(1, -1))
            return
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], int):
            r_idx = int(key[0])
            row_f = self.dequant_row(r_idx, out_dtype=torch.float32)
            row_f[key[1]] = value
            self.set_rows(torch.tensor([r_idx], device=self.device, dtype=torch.long), row_f.unsqueeze(0))
            return
        if isinstance(key, torch.Tensor) and key.ndim == 1:
            idx = torch.nonzero(key, as_tuple=False).squeeze(-1) if key.dtype == torch.bool else key
            val_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device, dtype=torch.float32)
            self.set_rows(idx, val_t)
            return
        f = self.float()
        f[key] = value
        self.copy_(f)

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        kwargs = kwargs or {}
        if func is torch.diag and len(args) == 1 and isinstance(args[0], QuantizedSynapseMatrix):
            return args[0].diagonal()
        if func is torch.isfinite and len(args) == 1 and isinstance(args[0], QuantizedSynapseMatrix):
            ok = args[0].is_all_finite()
            return torch.tensor(ok, device=args[0].device, dtype=torch.bool)
        if func is torch.mv and len(args) == 2 and isinstance(args[0], QuantizedSynapseMatrix):
            return args[0].matvec(args[1])
        # Fallback: convert QuantizedSynapseMatrix args to float32 tensors
        conv_args = tuple(a.float() if isinstance(a, QuantizedSynapseMatrix) else a for a in args)
        conv_kwargs = {k: (v.float() if isinstance(v, QuantizedSynapseMatrix) else v) for k, v in kwargs.items()}
        return func(*conv_args, **conv_kwargs)
