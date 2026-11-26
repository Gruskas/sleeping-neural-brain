from __future__ import annotations

import gc
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch
import torch.nn.functional as F

from .sensory_wernicke import k_wta


MIN_BUDGET_MB = 68.0
MAX_BUDGET_MB = 32768.0  # 32 GB


def trim_process_ram() -> None:
    """
    Reclaim unused CPU RAM pages (PyArrow pool + Python GC + Windows working-set trim).
    Zero impact on GPU VRAM tensors or CUDA throughput!
    """
    gc.collect()
    try:
        import sys as _sys
        if "pyarrow" in _sys.modules:
            _sys.modules["pyarrow"].default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        import os as _os
        if _os.name == "nt":
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessWorkingSetSize(h, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:
        pass


def _tensor_bytes(t: torch.Tensor | None) -> int:
    if t is None:
        return 0
    return int(t.numel() * t.element_size())


def _safe_k_for_size(n: int, ratio: float = 0.04) -> int:
    return max(8, min(n, int(round(n * ratio))))


@dataclass
class NeuromodulatorState:
    noradrenaline: float = 0.35   # Surprise / arousal / context reset [0.1 .. 2.5]
    dopamine: float = 0.50        # Valence / intrinsic curiosity reward [-0.5 .. 2.0]
    acetylcholine: float = 0.55   # Mode switch: high = M2 encode, low = M2 recall/replay [0.0 .. 1.0]

    def to_dict(self) -> dict[str, float]:
        return {
            "noradrenaline": round(self.noradrenaline, 3),
            "dopamine": round(self.dopamine, 3),
            "acetylcholine": round(self.acetylcholine, 3),
        }


@dataclass
class LivingStepTelemetry:
    surprise: float = 0.0
    surprise_l1: float = 0.0
    surprise_l2: float = 0.0
    surprise_l3: float = 0.0
    predictive_suppression_pct: float = 0.0
    intrinsic_curiosity: float = 0.0
    learning_progress: float = 0.0
    boredom_exploration: bool = False
    neurogenesis_events: int = 0
    neurons_added: int = 0
    pruned_synapses: int = 0
    pruned_episodes: int = 0
    m2_encoded: bool = False
    theta_phase: float = 0.0
    gamma_phase: float = 0.0
    mean_fatigue: float = 0.0
    neuromodulators: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surprise": round(self.surprise, 4),
            "surprise_l1": round(self.surprise_l1, 4),
            "surprise_l2": round(self.surprise_l2, 4),
            "surprise_l3": round(self.surprise_l3, 4),
            "predictive_suppression_pct": round(self.predictive_suppression_pct, 1),
            "intrinsic_curiosity": round(self.intrinsic_curiosity, 4),
            "learning_progress": round(self.learning_progress, 4),
            "boredom_exploration": self.boredom_exploration,
            "neurogenesis_events": self.neurogenesis_events,
            "neurons_added": self.neurons_added,
            "pruned_synapses": self.pruned_synapses,
            "pruned_episodes": self.pruned_episodes,
            "m2_encoded": self.m2_encoded,
            "theta_phase": round(self.theta_phase, 3),
            "gamma_phase": round(self.gamma_phase, 3),
            "mean_fatigue": round(self.mean_fatigue, 4),
            "neuromodulators": dict(self.neuromodulators),
        }


class LivingBrainEngine:
    """
    Biologically Inspired Dynamically Growing Cognitive Architecture ("Living Brain")
    built strictly on local learning rules inside `@torch.no_grad()` (ZERO backpropagation!):

      1. Unbounded Structural Neurogenesis (M1: L1, L2, L3, Assoc; M2: Hippocampus; M4: Basal Ganglia):
         - Zero hardcoded capacity limits; dynamically allocates new neurons & sparse synapses
           when prediction error persists, representations saturate, or novel patterns appear.
         - Balanced by priority pruning & Synaptic Homeostasis (SHY) during sleep & memory pressure.

      2. Hierarchical Predictive Coding & Surprise (L3 -> L2 -> L1):
         - Top-down predictions compared against bottom-up inputs.
         - Predictable signals undergo predictive suppression; high prediction error ("surprise")
           opens plasticity gates and routes novel episodes to M2 (Hippocampus).

      3. Intrinsic Curiosity & Cognitive Openness (M4 — Basal Ganglia):
         - Rewards moderate uncertainty + high learning progress (inverted-U Goldilocks curve),
           avoiding both boring loops and unlearnable noise.
         - When surprise stays too low, actively triggers epistemic exploration of under-explored
           memory regions and new associative links.

      4. Three-Factor Neuromodulated Oja-Hebb Plasticity:
         - Noradrenaline (NA): boosts plasticity & resets attention on sudden context shift.
         - Dopamine (DA): modulates eligibility traces based on M4 curiosity/valence & LP.
         - Acetylcholine (ACh): switches M2 continuously between fast encoding and recall/replay.
         - Offline replay (M2 -> M1) during sleep consolidates episodic traces into slow cortical weights.

      5. Continuous Spontaneous Activity & Homeostasis:
         - Background theta/gamma nested oscillations + k-WTA SDR sparsity (~4%) even without input.
         - Short-term firing-rate fatigue forces natural attention switching; long-term homeostatic
           thresholds stabilize firing rates.

      6. Dynamic Memory Budget Controller (68 MB .. 32 768 MB, live GPU VRAM / System RAM target):
         - Organic growth allowed while usage < 90% of budget (and safe physical hardware ceiling).
         - At 90-95% of budget: pauses net growth and triggers priority pruning & SHY homeostasis.
         - If slider drops below current usage: immediately runs controlled pruning & tensor compaction.
    """

    def __init__(
        self,
        sensory_dim: int = 512,
        target_device: str = "gpu",
        memory_budget_mb: float = 2048.0,
        start_background_clock: bool = False,
    ) -> None:
        self.lock = threading.RLock()
        self.sensory_dim = int(sensory_dim)
        self.target_device_mode = "gpu" if (target_device == "gpu" and torch.cuda.is_available()) else "ram"
        self.device = torch.device("cuda" if self.target_device_mode == "gpu" else "cpu")
        # Use compact bfloat16 on BOTH GPU and CPU RAM (2x lower RAM usage, zero GPU slowdown!)
        self.dtype = torch.bfloat16 if (self.device.type == "cpu" or torch.cuda.is_bf16_supported()) else torch.float16
        self.memory_budget_mb = max(MIN_BUDGET_MB, min(MAX_BUDGET_MB, float(memory_budget_mb)))

        # Biological floor sizes (allows compacting safely down to ~68 MB without breaking structure)
        self.min_l1 = 256
        self.min_l2 = 192
        self.min_l3 = 128
        self.min_assoc = 192
        self.min_m4 = 32

        # Dynamic layer sizes — NO hardcoded upper ceiling!
        self.n_l1 = 512
        self.n_l2 = 384
        self.n_l3 = 256
        self.n_assoc = 384
        self.n_m4 = 64

        self.neuromod = NeuromodulatorState()
        self.tick_count: int = 0
        self.theta_phase: float = 0.0
        self.gamma_phase: float = 0.0
        self.surprise_ema: float = 0.25
        self.prev_surprise: float = 0.25
        self.high_surprise_streak: int = 0
        self.low_surprise_streak: int = 0

        # Cumulative biological stats
        self.total_neurogenesis_events: int = 0
        self.total_neurons_born: int = 0
        self.total_neurons_pruned: int = 0
        self.total_synapses_pruned: int = 0
        self.total_episodes_pruned: int = 0
        self.total_replay_cycles: int = 0
        self.total_explorations: int = 0
        self.budget_status: str = "ROŚNIE"  # ROŚNIE | HOMEOSTAZA | KOMPAKTOWANIE
        self.last_telemetry = LivingStepTelemetry(neuromodulators=self.neuromod.to_dict())

        self._init_tensors()

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._bg_thread: threading.Thread | None = None
        if start_background_clock:
            self.start_spontaneous_clock()

    @torch.no_grad()
    def _init_tensors(self) -> None:
        dev = self.device
        dt = self.dtype

        def _sparse_mat(rows: int, cols: int, density: float = 0.12, scale: float = 0.15) -> torch.Tensor:
            w = torch.randn(rows, cols, device=dev, dtype=torch.float32) * scale
            mask = (torch.rand(rows, cols, device=dev) < density).float()
            w.mul_(mask)
            w = F.normalize(w, dim=-1, eps=1e-6) * scale
            return w.to(dtype=dt)

        # ── M1 Hierarchical Cortex (L1 <-> L2 <-> L3 <-> Assoc) ──────────────
        self.W_in_L1 = _sparse_mat(self.n_l1, self.sensory_dim, density=0.22, scale=0.25)
        self.G_10 = _sparse_mat(self.sensory_dim, self.n_l1, density=0.22, scale=0.20)

        self.W_12 = _sparse_mat(self.n_l2, self.n_l1, density=0.18, scale=0.22)
        self.G_21 = _sparse_mat(self.n_l1, self.n_l2, density=0.18, scale=0.20)

        self.W_23 = _sparse_mat(self.n_l3, self.n_l2, density=0.18, scale=0.22)
        self.G_32 = _sparse_mat(self.n_l2, self.n_l3, density=0.18, scale=0.20)

        self.W_3a = _sparse_mat(self.n_assoc, self.n_l3, density=0.20, scale=0.22)
        self.W_assoc_rec = _sparse_mat(self.n_assoc, self.n_assoc, density=0.10, scale=0.10)
        self.W_assoc_rec.fill_diagonal_(0.0)

        # Eligibility traces for Dopamine-gated 3-factor learning (compact float32 on key associative links)
        self.E_assoc_rec = torch.zeros((self.n_assoc, self.n_assoc), device=dev, dtype=dt)

        # Live neuron states, short-term fatigue (firing-rate adaptation), long-term homeostatic thresholds & utility
        self.state_l1 = torch.zeros(self.n_l1, device=dev, dtype=torch.float32)
        self.state_l2 = torch.zeros(self.n_l2, device=dev, dtype=torch.float32)
        self.state_l3 = torch.zeros(self.n_l3, device=dev, dtype=torch.float32)
        self.state_assoc = torch.zeros(self.n_assoc, device=dev, dtype=torch.float32)

        self.fatigue_l1 = torch.zeros(self.n_l1, device=dev, dtype=torch.float32)
        self.fatigue_l2 = torch.zeros(self.n_l2, device=dev, dtype=torch.float32)
        self.fatigue_l3 = torch.zeros(self.n_l3, device=dev, dtype=torch.float32)
        self.fatigue_assoc = torch.zeros(self.n_assoc, device=dev, dtype=torch.float32)

        self.thresh_l1 = torch.zeros(self.n_l1, device=dev, dtype=torch.float32)
        self.thresh_l2 = torch.zeros(self.n_l2, device=dev, dtype=torch.float32)
        self.thresh_l3 = torch.zeros(self.n_l3, device=dev, dtype=torch.float32)
        self.thresh_assoc = torch.zeros(self.n_assoc, device=dev, dtype=torch.float32)

        self.avg_act_l1 = torch.full((self.n_l1,), 0.04, device=dev, dtype=torch.float32)
        self.avg_act_l2 = torch.full((self.n_l2,), 0.04, device=dev, dtype=torch.float32)
        self.avg_act_l3 = torch.full((self.n_l3,), 0.04, device=dev, dtype=torch.float32)
        self.avg_act_assoc = torch.full((self.n_assoc,), 0.04, device=dev, dtype=torch.float32)

        # ── M2 Dynamic Episodic Hippocampus (Unbounded, budget-governed) ─────
        self.m2_assoc_traces = torch.empty((0, self.n_assoc), device=dev, dtype=dt)
        self.m2_sensory_traces = torch.empty((0, self.sensory_dim), device=dev, dtype=dt)
        self.m2_strengths = torch.empty((0,), device=dev, dtype=torch.float32)
        self.m2_surprises = torch.empty((0,), device=dev, dtype=torch.float32)
        self.m2_last_used = torch.empty((0,), device=dev, dtype=torch.int64)
        self.m2_valences = torch.empty((0,), device=dev, dtype=torch.float32)
        self.m2_pinned = torch.empty((0,), device=dev, dtype=torch.bool)

        # ── M4 Basal Ganglia (Intrinsic Curiosity, Striatal Prototypes & Valence)
        self.m4_prototypes = F.normalize(
            torch.randn(self.n_m4, self.n_assoc, device=dev, dtype=torch.float32), dim=-1
        ).to(dtype=dt)
        self.m4_visits = torch.zeros(self.n_m4, device=dev, dtype=torch.float32)
        self.m4_err_ema = torch.full((self.n_m4,), 0.45, device=dev, dtype=torch.float32)
        self.m4_values = torch.zeros(self.n_m4, device=dev, dtype=torch.float32)

    # ── Memory Accounting & Dynamic Budget Controller (Pillar 6) ─────────────
    def living_memory_bytes(self) -> int:
        tensors = [
            self.W_in_L1, self.G_10, self.W_12, self.G_21, self.W_23, self.G_32,
            self.W_3a, self.W_assoc_rec, self.E_assoc_rec,
            self.state_l1, self.state_l2, self.state_l3, self.state_assoc,
            self.fatigue_l1, self.fatigue_l2, self.fatigue_l3, self.fatigue_assoc,
            self.thresh_l1, self.thresh_l2, self.thresh_l3, self.thresh_assoc,
            self.avg_act_l1, self.avg_act_l2, self.avg_act_l3, self.avg_act_assoc,
            self.m2_assoc_traces, self.m2_sensory_traces,
            self.m2_strengths, self.m2_surprises, self.m2_last_used, self.m2_valences, self.m2_pinned,
            self.m4_prototypes, self.m4_visits, self.m4_err_ema, self.m4_values,
        ]
        return sum(_tensor_bytes(t) for t in tensors)

    def living_memory_mb(self) -> float:
        return round(self.living_memory_bytes() / (1024.0 * 1024.0), 2)

    def hardware_memory_stats(self) -> dict[str, float]:
        proc_rss_mb = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 1)
        sys_total_mb = round(psutil.virtual_memory().total / (1024.0 * 1024.0), 1)
        if torch.cuda.is_available():
            gpu_alloc_mb = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 1)
            gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0), 1)
        else:
            gpu_alloc_mb = 0.0
            gpu_total_mb = 0.0
        return {
            "living_mb": self.living_memory_mb(),
            "gpu_allocated_mb": gpu_alloc_mb,
            "gpu_total_mb": gpu_total_mb,
            "ram_rss_mb": proc_rss_mb,
            "ram_total_mb": sys_total_mb,
        }

    def effective_budget_mb(self) -> float:
        """
        Effective ceiling honoring the user's slider (68 MB .. 32 768 MB) while ALSO
        guarding against physical hardware OOM on GPU (leaving headroom for PyTorch allocator).
        """
        hw = self.hardware_memory_stats()
        if self.target_device_mode == "gpu" and hw["gpu_total_mb"] > 0:
            # Physical GPU headroom: keep total GPU allocation safely under 92% of physical VRAM
            other_gpu_mb = max(0.0, hw["gpu_allocated_mb"] - hw["living_mb"])
            max_safe_living_gpu = max(MIN_BUDGET_MB, hw["gpu_total_mb"] * 0.90 - other_gpu_mb)
            return round(min(self.memory_budget_mb, max_safe_living_gpu), 2)
        if hw["ram_total_mb"] > 0:
            max_safe_ram = max(MIN_BUDGET_MB, hw["ram_total_mb"] * 0.85)
            return round(min(self.memory_budget_mb, max_safe_ram), 2)
        return round(self.memory_budget_mb, 2)

    def budget_utilization_ratio(self) -> float:
        eff = max(MIN_BUDGET_MB, self.effective_budget_mb())
        return self.living_memory_mb() / eff

    def active_synapses_count(self) -> int:
        with self.lock:
            return int(
                (self.W_in_L1.abs() > 0).sum().item()
                + (self.W_12.abs() > 0).sum().item()
                + (self.W_23.abs() > 0).sum().item()
                + (self.W_3a.abs() > 0).sum().item()
                + (self.W_assoc_rec.abs() > 0).sum().item()
            )

    def total_neurons_count(self) -> int:
        return int(self.n_l1 + self.n_l2 + self.n_l3 + self.n_assoc + self.n_m4)

    @torch.no_grad()
    def set_memory_budget(
        self,
        budget_mb: float | None = None,
        target_device: str | None = None,
    ) -> dict[str, Any]:
        """
        Live UI slider & device selector handler (68 MB .. 32 768 MB, GPU VRAM or System RAM).
        If the user lowers the slider below current usage, immediately runs controlled pruning
        and tensor compaction so the brain safely shrinks to the new limit without OOM!
        """
        with self.lock:
            if target_device in ("gpu", "ram"):
                want_gpu = (target_device == "gpu") and torch.cuda.is_available()
                new_mode = "gpu" if want_gpu else "ram"
                if new_mode != self.target_device_mode:
                    self._migrate_device(new_mode)

            if budget_mb is not None:
                self.memory_budget_mb = max(MIN_BUDGET_MB, min(MAX_BUDGET_MB, float(budget_mb)))

            compacted = self.enforce_memory_budget()
            return {
                "memory_budget_mb": round(self.memory_budget_mb, 1),
                "effective_budget_mb": self.effective_budget_mb(),
                "target_device": self.target_device_mode,
                "living_mb": self.living_memory_mb(),
                "budget_status": self.budget_status,
                **compacted,
            }

    @torch.no_grad()
    def _migrate_device(self, new_mode: str) -> None:
        new_dev = torch.device("cuda" if (new_mode == "gpu" and torch.cuda.is_available()) else "cpu")
        new_dt = torch.bfloat16 if (new_dev.type == "cpu" or torch.cuda.is_bf16_supported()) else torch.float16
        mat_names = [
            "W_in_L1", "G_10", "W_12", "G_21", "W_23", "G_32",
            "W_3a", "W_assoc_rec", "E_assoc_rec",
            "m2_assoc_traces", "m2_sensory_traces", "m4_prototypes",
        ]
        vec_names = [
            "state_l1", "state_l2", "state_l3", "state_assoc",
            "fatigue_l1", "fatigue_l2", "fatigue_l3", "fatigue_assoc",
            "thresh_l1", "thresh_l2", "thresh_l3", "thresh_assoc",
            "avg_act_l1", "avg_act_l2", "avg_act_l3", "avg_act_assoc",
            "m2_strengths", "m2_surprises", "m2_last_used", "m2_valences", "m2_pinned",
            "m4_visits", "m4_err_ema", "m4_values",
        ]
        for name in mat_names:
            setattr(self, name, getattr(self, name).to(device=new_dev, dtype=new_dt).contiguous())
        for name in vec_names:
            setattr(self, name, getattr(self, name).to(device=new_dev).contiguous())
        self.target_device_mode = new_mode
        self.device = new_dev
        self.dtype = new_dt
        trim_process_ram()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def enforce_memory_budget(self) -> dict[str, int]:
        """
        Enforce Pillar 6 dynamic memory policy:
          - < 90% budget: status = ROŚNIE (organic growth enabled)
          - 90% .. 100% budget: status = HOMEOSTAZA (pause growth, run priority SHY + M2 pruning)
          - > 100% budget (e.g. slider dragged down): status = KOMPAKTOWANIE (immediate controlled
            pruning & tensor compaction of M2 episodes + least-active M1/M4 neurons until under limit!)
        """
        pruned_syn = 0
        pruned_ep = 0
        pruned_neu = 0

        ratio = self.budget_utilization_ratio()
        if ratio < 0.90:
            self.budget_status = "ROŚNIE"
            return {"pruned_synapses": 0, "pruned_episodes": 0, "pruned_neurons": 0}

        if ratio <= 1.0:
            self.budget_status = "HOMEOSTAZA"
            pruned_syn += self._prune_weak_synapses(eps=0.015)
            pruned_ep += self._prune_weak_episodes(fraction=0.15)
            return {"pruned_synapses": pruned_syn, "pruned_episodes": pruned_ep, "pruned_neurons": 0}

        # Over budget (> 100%): Immediate controlled compaction!
        self.budget_status = "KOMPAKTOWANIE"
        for _ in range(16):
            if self.budget_utilization_ratio() <= 0.92:
                break
            # 1) First prune weakest unpinned M2 episodes
            ep_n = self._prune_weak_episodes(fraction=0.35)
            pruned_ep += ep_n
            # 2) Prune weak M1 synapses
            pruned_syn += self._prune_weak_synapses(eps=0.025)
            if self.budget_utilization_ratio() <= 0.92:
                break
            # 3) Compact least-active neurons in M1 (L1/L2/L3/Assoc) and M4 down toward biological floor
            n_removed = self._compact_least_active_neurons(shrink_factor=0.78)
            pruned_neu += n_removed
            if ep_n == 0 and n_removed == 0:
                break

        trim_process_ram()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        ratio_after = self.budget_utilization_ratio()
        self.budget_status = "ROŚNIE" if ratio_after < 0.90 else ("HOMEOSTAZA" if ratio_after <= 1.0 else "KOMPAKTOWANIE")
        return {
            "pruned_synapses": pruned_syn,
            "pruned_episodes": pruned_ep,
            "pruned_neurons": pruned_neu,
        }

    @torch.no_grad()
    def _prune_weak_synapses(self, eps: float = 0.012, downscale: float = 0.96) -> int:
        pruned = 0
        for name in ("W_in_L1", "W_12", "W_23", "W_3a", "W_assoc_rec"):
            w = getattr(self, name)
            w.mul_(downscale)
            mask = (w.abs() < eps) & (w != 0)
            c = int(mask.sum().item())
            if c > 0:
                w[mask] = 0
                pruned += c
        self.W_assoc_rec.fill_diagonal_(0.0)
        self.total_synapses_pruned += pruned
        return pruned

    @torch.no_grad()
    def _prune_weak_episodes(self, fraction: float = 0.20) -> int:
        m = int(self.m2_assoc_traces.shape[0])
        if m == 0:
            return 0
        pinned_idx = torch.nonzero(self.m2_pinned, as_tuple=False).squeeze(-1)
        unpinned_idx = torch.nonzero(~self.m2_pinned, as_tuple=False).squeeze(-1)
        u_count = int(unpinned_idx.numel())
        if u_count == 0:
            return 0
        if m <= 8 and int(pinned_idx.numel()) == 0:
            return 0
        drop_n = max(1, int(round(u_count * fraction)))
        min_unpinned = 0 if (int(pinned_idx.numel()) > 0 and fraction >= 0.30) else min(4, u_count)
        keep_unpinned_n = max(min_unpinned, u_count - drop_n)
        if keep_unpinned_n >= u_count:
            return 0
        # Retention depends on strength, surprise, and valence — never on age/recency
        priority = self.m2_strengths + 0.6 * self.m2_surprises + 0.3 * self.m2_valences
        if keep_unpinned_n > 0:
            u_prio = priority.index_select(0, unpinned_idx)
            top_rel = torch.topk(u_prio, k=keep_unpinned_n, largest=True, sorted=False).indices
            keep_u = unpinned_idx.index_select(0, top_rel)
            keep_idx = torch.cat([pinned_idx, keep_u], dim=0)
        else:
            keep_idx = pinned_idx
        keep_idx, _ = torch.sort(keep_idx)
        self.m2_assoc_traces = self.m2_assoc_traces.index_select(0, keep_idx).contiguous()
        self.m2_sensory_traces = self.m2_sensory_traces.index_select(0, keep_idx).contiguous()
        self.m2_strengths = self.m2_strengths.index_select(0, keep_idx).contiguous()
        self.m2_surprises = self.m2_surprises.index_select(0, keep_idx).contiguous()
        self.m2_last_used = self.m2_last_used.index_select(0, keep_idx).contiguous()
        self.m2_valences = self.m2_valences.index_select(0, keep_idx).contiguous()
        self.m2_pinned = self.m2_pinned.index_select(0, keep_idx).contiguous()
        removed = m - int(keep_idx.numel())
        self.total_episodes_pruned += removed
        return removed

    @torch.no_grad()
    def _compact_least_active_neurons(self, shrink_factor: float = 0.80) -> int:
        removed_total = 0

        def _select_keep(activity: torch.Tensor, w_row: torch.Tensor, cur_n: int, min_n: int) -> torch.Tensor | None:
            if cur_n <= min_n:
                return None
            new_n = max(min_n, int(round(cur_n * shrink_factor)))
            if new_n >= cur_n:
                return None
            utility = activity + 0.5 * w_row.float().abs().mean(dim=-1)
            idx = torch.topk(utility, k=new_n, largest=True, sorted=False).indices
            idx, _ = torch.sort(idx)
            return idx

        # Compact L1
        idx1 = _select_keep(self.avg_act_l1, self.W_in_L1, self.n_l1, self.min_l1)
        if idx1 is not None:
            removed_total += self.n_l1 - int(idx1.numel())
            self.n_l1 = int(idx1.numel())
            self.W_in_L1 = self.W_in_L1.index_select(0, idx1).contiguous()
            self.G_10 = self.G_10.index_select(1, idx1).contiguous()
            self.W_12 = self.W_12.index_select(1, idx1).contiguous()
            self.G_21 = self.G_21.index_select(0, idx1).contiguous()
            self.state_l1 = self.state_l1.index_select(0, idx1).contiguous()
            self.fatigue_l1 = self.fatigue_l1.index_select(0, idx1).contiguous()
            self.thresh_l1 = self.thresh_l1.index_select(0, idx1).contiguous()
            self.avg_act_l1 = self.avg_act_l1.index_select(0, idx1).contiguous()

        # Compact L2
        idx2 = _select_keep(self.avg_act_l2, self.W_12, self.n_l2, self.min_l2)
        if idx2 is not None:
            removed_total += self.n_l2 - int(idx2.numel())
            self.n_l2 = int(idx2.numel())
            self.W_12 = self.W_12.index_select(0, idx2).contiguous()
            self.G_21 = self.G_21.index_select(1, idx2).contiguous()
            self.W_23 = self.W_23.index_select(1, idx2).contiguous()
            self.G_32 = self.G_32.index_select(0, idx2).contiguous()
            self.state_l2 = self.state_l2.index_select(0, idx2).contiguous()
            self.fatigue_l2 = self.fatigue_l2.index_select(0, idx2).contiguous()
            self.thresh_l2 = self.thresh_l2.index_select(0, idx2).contiguous()
            self.avg_act_l2 = self.avg_act_l2.index_select(0, idx2).contiguous()

        # Compact L3
        idx3 = _select_keep(self.avg_act_l3, self.W_23, self.n_l3, self.min_l3)
        if idx3 is not None:
            removed_total += self.n_l3 - int(idx3.numel())
            self.n_l3 = int(idx3.numel())
            self.W_23 = self.W_23.index_select(0, idx3).contiguous()
            self.G_32 = self.G_32.index_select(1, idx3).contiguous()
            self.W_3a = self.W_3a.index_select(1, idx3).contiguous()
            self.state_l3 = self.state_l3.index_select(0, idx3).contiguous()
            self.fatigue_l3 = self.fatigue_l3.index_select(0, idx3).contiguous()
            self.thresh_l3 = self.thresh_l3.index_select(0, idx3).contiguous()
            self.avg_act_l3 = self.avg_act_l3.index_select(0, idx3).contiguous()

        # Compact Associative Hub (and aligned M2/M4 columns, protecting neurons carrying pinned M2 traces!)
        assoc_activity = self.avg_act_assoc
        if self.m2_assoc_traces.shape[0] > 0 and bool(self.m2_pinned.any().item()):
            pin_imp = self.m2_assoc_traces[self.m2_pinned].float().abs().mean(dim=0)
            assoc_activity = assoc_activity + 1.5 * pin_imp
        idxa = _select_keep(assoc_activity, self.W_3a, self.n_assoc, self.min_assoc)
        if idxa is not None:
            removed_total += self.n_assoc - int(idxa.numel())
            self.n_assoc = int(idxa.numel())
            self.W_3a = self.W_3a.index_select(0, idxa).contiguous()
            self.W_assoc_rec = self.W_assoc_rec.index_select(0, idxa).index_select(1, idxa).contiguous()
            self.W_assoc_rec.fill_diagonal_(0.0)
            self.E_assoc_rec = self.E_assoc_rec.index_select(0, idxa).index_select(1, idxa).contiguous()
            self.state_assoc = self.state_assoc.index_select(0, idxa).contiguous()
            self.fatigue_assoc = self.fatigue_assoc.index_select(0, idxa).contiguous()
            self.thresh_assoc = self.thresh_assoc.index_select(0, idxa).contiguous()
            self.avg_act_assoc = self.avg_act_assoc.index_select(0, idxa).contiguous()
            if self.m2_assoc_traces.shape[0] > 0:
                self.m2_assoc_traces = F.normalize(
                    self.m2_assoc_traces.index_select(1, idxa).float(), dim=-1, eps=1e-6
                ).to(dtype=self.dtype).contiguous()
            else:
                self.m2_assoc_traces = torch.empty((0, self.n_assoc), device=self.device, dtype=self.dtype)
            self.m4_prototypes = F.normalize(
                self.m4_prototypes.index_select(1, idxa).float(), dim=-1, eps=1e-6
            ).to(dtype=self.dtype).contiguous()

        # Compact M4 Basal Ganglia units
        if self.n_m4 > self.min_m4:
            new_m4 = max(self.min_m4, int(round(self.n_m4 * shrink_factor)))
            if new_m4 < self.n_m4:
                keep_m4 = torch.topk(self.m4_visits + self.m4_values.abs(), k=new_m4, largest=True, sorted=False).indices
                keep_m4, _ = torch.sort(keep_m4)
                removed_total += self.n_m4 - new_m4
                self.n_m4 = new_m4
                self.m4_prototypes = self.m4_prototypes.index_select(0, keep_m4).contiguous()
                self.m4_visits = self.m4_visits.index_select(0, keep_m4).contiguous()
                self.m4_err_ema = self.m4_err_ema.index_select(0, keep_m4).contiguous()
                self.m4_values = self.m4_values.index_select(0, keep_m4).contiguous()

        self.total_neurons_pruned += removed_total
        return removed_total

    # ── Pillar 1: Structural Neurogenesis (Unbounded Organic Growth!) ────────
    @torch.no_grad()
    def grow_structure(
        self,
        add_l1: int = 128,
        add_l2: int = 96,
        add_l3: int = 64,
        add_assoc: int = 96,
        add_m4: int = 32,
        seed_input: torch.Tensor | None = None,
    ) -> int:
        """
        Real structural neurogenesis: dynamically allocates new neurons & sparse synapses
        across M1 (L1, L2, L3, Assoc) and M4 (Basal Ganglia), gated ONLY by the dynamic
        memory budget (< 90% of slider limit and physical hardware ceiling).
        """
        with self.lock:
            if self.budget_utilization_ratio() >= 0.90:
                self.enforce_memory_budget()
                return 0

            dev = self.device
            dt = self.dtype

            def _sparse_block(rows: int, cols: int, density: float = 0.14, scale: float = 0.16) -> torch.Tensor:
                w = torch.randn(rows, cols, device=dev, dtype=torch.float32) * scale
                mask = (torch.rand(rows, cols, device=dev) < density).float()
                w.mul_(mask)
                return w.to(dtype=dt)

            born = 0

            # Grow L1
            if add_l1 > 0:
                new_in = _sparse_block(add_l1, self.sensory_dim, density=0.22, scale=0.22)
                if seed_input is not None and seed_input.numel() == self.sensory_dim:
                    # Bias newborn L1 receptive fields gently toward the surprising stimulus!
                    s_vec = F.normalize(seed_input.to(device=dev, dtype=torch.float32).reshape(1, -1), dim=-1)
                    new_in = F.normalize(new_in.float() + 0.35 * s_vec, dim=-1, eps=1e-6).to(dtype=dt) * 0.25
                self.W_in_L1 = torch.cat([self.W_in_L1, new_in], dim=0).contiguous()
                self.G_10 = torch.cat([self.G_10, _sparse_block(self.sensory_dim, add_l1, 0.20, 0.18)], dim=1).contiguous()
                self.W_12 = torch.cat([self.W_12, _sparse_block(self.n_l2, add_l1, 0.16, 0.18)], dim=1).contiguous()
                self.G_21 = torch.cat([self.G_21, _sparse_block(add_l1, self.n_l2, 0.16, 0.18)], dim=0).contiguous()
                self.state_l1 = torch.cat([self.state_l1, torch.zeros(add_l1, device=dev)], dim=0)
                self.fatigue_l1 = torch.cat([self.fatigue_l1, torch.zeros(add_l1, device=dev)], dim=0)
                self.thresh_l1 = torch.cat([self.thresh_l1, torch.zeros(add_l1, device=dev)], dim=0)
                self.avg_act_l1 = torch.cat([self.avg_act_l1, torch.full((add_l1,), 0.04, device=dev)], dim=0)
                self.n_l1 += add_l1
                born += add_l1

            # Grow L2
            if add_l2 > 0:
                self.W_12 = torch.cat([self.W_12, _sparse_block(add_l2, self.n_l1, 0.16, 0.20)], dim=0).contiguous()
                self.G_21 = torch.cat([self.G_21, _sparse_block(self.n_l1, add_l2, 0.16, 0.18)], dim=1).contiguous()
                self.W_23 = torch.cat([self.W_23, _sparse_block(self.n_l3, add_l2, 0.16, 0.18)], dim=1).contiguous()
                self.G_32 = torch.cat([self.G_32, _sparse_block(add_l2, self.n_l3, 0.16, 0.18)], dim=0).contiguous()
                self.state_l2 = torch.cat([self.state_l2, torch.zeros(add_l2, device=dev)], dim=0)
                self.fatigue_l2 = torch.cat([self.fatigue_l2, torch.zeros(add_l2, device=dev)], dim=0)
                self.thresh_l2 = torch.cat([self.thresh_l2, torch.zeros(add_l2, device=dev)], dim=0)
                self.avg_act_l2 = torch.cat([self.avg_act_l2, torch.full((add_l2,), 0.04, device=dev)], dim=0)
                self.n_l2 += add_l2
                born += add_l2

            # Grow L3
            if add_l3 > 0:
                self.W_23 = torch.cat([self.W_23, _sparse_block(add_l3, self.n_l2, 0.16, 0.20)], dim=0).contiguous()
                self.G_32 = torch.cat([self.G_32, _sparse_block(self.n_l2, add_l3, 0.16, 0.18)], dim=1).contiguous()
                self.W_3a = torch.cat([self.W_3a, _sparse_block(self.n_assoc, add_l3, 0.18, 0.18)], dim=1).contiguous()
                self.state_l3 = torch.cat([self.state_l3, torch.zeros(add_l3, device=dev)], dim=0)
                self.fatigue_l3 = torch.cat([self.fatigue_l3, torch.zeros(add_l3, device=dev)], dim=0)
                self.thresh_l3 = torch.cat([self.thresh_l3, torch.zeros(add_l3, device=dev)], dim=0)
                self.avg_act_l3 = torch.cat([self.avg_act_l3, torch.full((add_l3,), 0.04, device=dev)], dim=0)
                self.n_l3 += add_l3
                born += add_l3

            # Grow Associative Hub + expand aligned M2 episodic columns & M4 prototype columns
            if add_assoc > 0:
                old_a = self.n_assoc
                new_a = old_a + add_assoc
                self.W_3a = torch.cat([self.W_3a, _sparse_block(add_assoc, self.n_l3, 0.18, 0.20)], dim=0).contiguous()
                new_rec = torch.zeros((new_a, new_a), device=dev, dtype=dt)
                new_rec[:old_a, :old_a] = self.W_assoc_rec
                new_rec[old_a:, :old_a] = _sparse_block(add_assoc, old_a, 0.08, 0.08)
                new_rec[:old_a, old_a:] = _sparse_block(old_a, add_assoc, 0.08, 0.08)
                new_rec[old_a:, old_a:] = _sparse_block(add_assoc, add_assoc, 0.08, 0.08)
                new_rec.fill_diagonal_(0.0)
                self.W_assoc_rec = new_rec.contiguous()

                new_etrace = torch.zeros((new_a, new_a), device=dev, dtype=dt)
                new_etrace[:old_a, :old_a] = self.E_assoc_rec
                self.E_assoc_rec = new_etrace.contiguous()

                self.state_assoc = torch.cat([self.state_assoc, torch.zeros(add_assoc, device=dev)], dim=0)
                self.fatigue_assoc = torch.cat([self.fatigue_assoc, torch.zeros(add_assoc, device=dev)], dim=0)
                self.thresh_assoc = torch.cat([self.thresh_assoc, torch.zeros(add_assoc, device=dev)], dim=0)
                self.avg_act_assoc = torch.cat([self.avg_act_assoc, torch.full((add_assoc,), 0.04, device=dev)], dim=0)

                if self.m2_assoc_traces.shape[0] > 0:
                    pad_m2 = torch.zeros((self.m2_assoc_traces.shape[0], add_assoc), device=dev, dtype=dt)
                    self.m2_assoc_traces = torch.cat([self.m2_assoc_traces, pad_m2], dim=1).contiguous()
                else:
                    self.m2_assoc_traces = torch.empty((0, new_a), device=dev, dtype=dt)

                pad_m4 = torch.zeros((self.n_m4, add_assoc), device=dev, dtype=dt)
                self.m4_prototypes = torch.cat([self.m4_prototypes, pad_m4], dim=1).contiguous()

                self.n_assoc = new_a
                born += add_assoc

            # Grow M4 Basal Ganglia striatal units
            if add_m4 > 0:
                new_proto = F.normalize(
                    torch.randn(add_m4, self.n_assoc, device=dev, dtype=torch.float32), dim=-1
                ).to(dtype=dt)
                self.m4_prototypes = torch.cat([self.m4_prototypes, new_proto], dim=0).contiguous()
                self.m4_visits = torch.cat([self.m4_visits, torch.zeros(add_m4, device=dev)], dim=0)
                self.m4_err_ema = torch.cat([self.m4_err_ema, torch.full((add_m4,), 0.45, device=dev)], dim=0)
                self.m4_values = torch.cat([self.m4_values, torch.zeros(add_m4, device=dev)], dim=0)
                self.n_m4 += add_m4
                born += add_m4

            if born > 0:
                self.total_neurogenesis_events += 1
                self.total_neurons_born += born
            self.enforce_memory_budget()
            return born

    # ── Core Biological Cognitive Step (Pillars 2, 3, 4, 5 — ZERO Backprop!) ─
    @torch.no_grad()
    def _project_sensory(self, sensory_vec: torch.Tensor | None) -> torch.Tensor:
        if sensory_vec is None:
            # Spontaneous background sensory drive modulated by theta/gamma oscillations
            osc = 0.55 + 0.30 * math.sin(self.theta_phase) + 0.15 * math.sin(self.gamma_phase)
            raw = torch.randn(self.sensory_dim, device=self.device, dtype=torch.float32) * (0.20 * osc)
            if self.state_l1.abs().max().item() > 1e-6:
                raw = raw + 0.65 * (self.G_10.float() @ self.state_l1)
            return F.normalize(raw, dim=-1, eps=1e-6)

        v = sensory_vec.to(device=self.device, dtype=torch.float32).reshape(-1)
        if v.numel() == self.sensory_dim:
            return F.normalize(v, dim=-1, eps=1e-6)
        # Adaptive deterministic folding into sensory_dim (no fixed input dim mismatch!)
        if v.numel() > self.sensory_dim:
            # Chunk-average folding preserves all input dimensions
            pad = (-v.numel()) % self.sensory_dim
            if pad > 0:
                v = F.pad(v, (0, pad))
            folded = v.view(-1, self.sensory_dim).mean(dim=0)
            return F.normalize(folded, dim=-1, eps=1e-6)
        padded = F.pad(v, (0, self.sensory_dim - v.numel()))
        return F.normalize(padded, dim=-1, eps=1e-6)

    @torch.no_grad()
    def get_cortical_association_vector(self, top_k_m2: int = 3) -> torch.Tensor:
        """
        Build a size-independent associative readout combining live `state_assoc`
        with the softmax-weighted average of the top-k (default 3) most strongly
        activated episodic traces from `m2_assoc_traces` for direct Z* projection.
        """
        with self.lock:
            state = self.state_assoc.detach().float()
            state_norm = float(state.norm().item())
            has_state = state_norm > 1e-6
            state_unit = (state / state_norm) if has_state else torch.zeros_like(state)

            m = int(self.m2_assoc_traces.shape[0])
            if m == 0:
                return state_unit

            m2_f = self.m2_assoc_traces.detach().float()
            if has_state:
                sims = m2_f @ state_unit
            else:
                sims = self.m2_strengths.detach().float()

            k = min(max(1, int(top_k_m2)), m)
            top_vals, top_idx = torch.topk(sims, k=k, largest=True, sorted=True)
            weights = F.softmax(6.0 * top_vals, dim=0)
            top_traces = m2_f.index_select(0, top_idx)
            m2_weighted = (weights.unsqueeze(1) * top_traces).sum(dim=0)
            m2_weighted = F.normalize(m2_weighted, dim=-1, eps=1e-6)

            if not has_state:
                return m2_weighted
            return F.normalize(0.65 * state_unit + 0.35 * m2_weighted, dim=-1, eps=1e-6)

    @torch.no_grad()
    def cognitive_step(
        self,
        sensory_vec: torch.Tensor | None = None,
        external_valence: float = 0.0,
        allow_growth: bool = True,
        spontaneous: bool = False,
        pinned: bool = False,
    ) -> LivingStepTelemetry:
        """
        Execute one full local-plasticity Living Brain cycle (ZERO backprop!):
          1. Theta/Gamma clock tick + short-term firing-rate fatigue recovery.
          2. Hierarchical Predictive Coding (L3 -> L2 -> L1) + Predictive Suppression.
          3. M4 Basal Ganglia Intrinsic Curiosity (Goldilocks uncertainty + learning progress)
             + Boredom-triggered epistemic exploration of under-explored memory regions.
          4. Neuromodulators update (Noradrenaline, Dopamine, Acetylcholine).
          5. 3-Factor Oja-Hebb local synaptic plasticity + M2 episodic encoding (gated by ACh & surprise).
          6. Structural Neurogenesis if surprise persists / layers saturate (within memory budget).
        """
        with self.lock:
            self.tick_count += 1
            # Nested Theta (~6 Hz) and Gamma (~40 Hz) oscillators
            self.theta_phase = (self.theta_phase + 0.35) % (2.0 * math.pi)
            self.gamma_phase = (self.gamma_phase + 2.30) % (2.0 * math.pi)
            theta_gate = 0.85 + 0.15 * math.sin(self.theta_phase)
            gamma_mod = 0.04 * math.sin(self.gamma_phase)

            x = self._project_sensory(sensory_vec)

            # ── Pillar 2: Top-Down Predictions (L3 -> L2 -> L1 -> Sensory) ───
            def _norm_pe(target: torch.Tensor, pred: torch.Tensor) -> float:
                if float(target.norm().item()) < 1e-6:
                    return 0.0
                if float(pred.norm().item()) < 1e-6:
                    return 0.85
                cos = float(F.cosine_similarity(target.unsqueeze(0), pred.unsqueeze(0)).item())
                return max(0.0, min(1.0, 0.5 * (1.0 - cos)))

            pred_l2 = F.normalize(self.G_32.float() @ self.state_l3, dim=-1, eps=1e-6) if self.state_l3.abs().max() > 1e-6 else torch.zeros_like(self.state_l2)
            pred_l1 = F.normalize(self.G_21.float() @ self.state_l2, dim=-1, eps=1e-6) if self.state_l2.abs().max() > 1e-6 else torch.zeros_like(self.state_l1)
            pred_x = F.normalize(self.G_10.float() @ self.state_l1, dim=-1, eps=1e-6) if self.state_l1.abs().max() > 1e-6 else torch.zeros_like(x)

            # Bottom-up prediction error & Scale-Invariant Predictive Suppression
            err_x = x - pred_x
            surprise_x = _norm_pe(x, pred_x)
            err_scale = err_x.abs().mean().clamp(min=1e-4)
            sup_mask_x = torch.sigmoid(((err_x.abs() / err_scale) - 0.65) * 4.0)
            suppressed_err_x = err_x * sup_mask_x
            suppression_pct = float((sup_mask_x < 0.45).float().mean().item() * 100.0)

            # ── Bottom-up + Top-down k-WTA with Fatigue & Homeostasis (Pillar 5)
            # Layer L1
            raw_l1 = (self.W_in_L1.float() @ (0.55 * x + 0.45 * suppressed_err_x)) * theta_gate
            err_l1 = raw_l1 - 0.35 * pred_l1
            surprise_l1 = _norm_pe(raw_l1, pred_l1)
            pre_l1 = raw_l1 + 0.25 * err_l1 - self.thresh_l1 - self.fatigue_l1 + gamma_mod
            z_l1, _, _ = k_wta(pre_l1.unsqueeze(0), k=_safe_k_for_size(self.n_l1), normalize=True)
            z_l1 = z_l1.squeeze(0)

            # Layer L2
            raw_l2 = (self.W_12.float() @ z_l1) * theta_gate
            err_l2 = raw_l2 - 0.35 * pred_l2
            surprise_l2 = _norm_pe(raw_l2, pred_l2)
            pre_l2 = raw_l2 + 0.25 * err_l2 - self.thresh_l2 - self.fatigue_l2 + gamma_mod
            z_l2, _, _ = k_wta(pre_l2.unsqueeze(0), k=_safe_k_for_size(self.n_l2), normalize=True)
            z_l2 = z_l2.squeeze(0)

            # Layer L3
            raw_l3 = (self.W_23.float() @ z_l2) * theta_gate
            pre_l3 = raw_l3 - self.thresh_l3 - self.fatigue_l3 + gamma_mod
            z_l3, _, _ = k_wta(pre_l3.unsqueeze(0), k=_safe_k_for_size(self.n_l3), normalize=True)
            z_l3 = z_l3.squeeze(0)

            # Associative Hub (M1_assoc) integrating L3 + recurrent W_assoc_rec + M2 recall when ACh is low
            h_rec = self.W_assoc_rec.float() @ self.state_assoc
            m2_recall_drive = torch.zeros_like(self.state_assoc)
            if self.m2_assoc_traces.shape[0] > 0 and self.neuromod.acetylcholine < 0.75:
                q_a = F.normalize(self.W_3a.float() @ z_l3, dim=-1, eps=1e-6)
                sims = self.m2_assoc_traces.float() @ q_a
                attn = F.softmax(10.0 * sims, dim=0)
                m2_recall_drive = F.normalize(attn @ self.m2_assoc_traces.float(), dim=-1, eps=1e-6)
                top_i = int(torch.argmax(sims).item())
                self.m2_last_used[top_i] = self.tick_count

            pre_assoc = (
                0.70 * (self.W_3a.float() @ z_l3)
                + 0.20 * h_rec
                + (1.0 - self.neuromod.acetylcholine) * 0.22 * m2_recall_drive
                - self.thresh_assoc
                - self.fatigue_assoc
            )
            z_assoc, _, _ = k_wta(pre_assoc.unsqueeze(0), k=_safe_k_for_size(self.n_assoc), normalize=True)
            z_assoc = z_assoc.squeeze(0)

            total_surprise = float(0.45 * surprise_x + 0.30 * surprise_l1 + 0.25 * surprise_l2)
            learning_progress = max(0.0, self.prev_surprise - total_surprise)
            self.prev_surprise = total_surprise

            # ── Pillar 3: M4 Basal Ganglia — Intrinsic Curiosity & Openness ──
            m4_sims = self.m4_prototypes.float() @ z_assoc  # [n_m4]
            best_m4_idx = int(torch.argmax(m4_sims).item())
            best_m4_sim = float(m4_sims[best_m4_idx].item())
            uncertainty = max(0.0, min(1.0, 1.0 - max(0.0, best_m4_sim)))

            prev_unit_err = float(self.m4_err_ema[best_m4_idx].item())
            unit_lp = max(0.0, prev_unit_err - total_surprise)
            self.m4_err_ema[best_m4_idx] = 0.85 * prev_unit_err + 0.15 * total_surprise
            self.m4_visits[best_m4_idx] += 1.0

            # Goldilocks / inverted-U curiosity curve:
            # Peak reward at moderate uncertainty (~0.45) + high learning progress;
            # penalizes boring repetitive loops (uncertainty < 0.15) AND pure unlearnable noise (> 0.85, LP ~ 0)
            goldilocks = math.exp(-(((uncertainty - 0.45) / 0.26) ** 2))
            intrinsic_curiosity = float(goldilocks * (0.45 + 2.5 * (learning_progress + unit_lp)))

            boredom_exploration = False
            if total_surprise < 0.16:
                self.low_surprise_streak += 1
            else:
                self.low_surprise_streak = 0

            # Force cognitive openness when surprise stays too low: explore least-visited memory region!
            if self.low_surprise_streak >= 3:
                boredom_exploration = True
                self.total_explorations += 1
                self.low_surprise_streak = 0
                least_visited = int(torch.argmin(self.m4_visits).item())
                explore_vec = self.m4_prototypes[least_visited].float()
                if self.m2_assoc_traces.shape[0] >= 2:
                    oldest_idx = int(torch.argmin(self.m2_last_used).item())
                    explore_vec = F.normalize(0.6 * explore_vec + 0.4 * self.m2_assoc_traces[oldest_idx].float(), dim=-1)
                z_assoc, _, _ = k_wta((0.55 * z_assoc + 0.45 * explore_vec).unsqueeze(0), k=_safe_k_for_size(self.n_assoc), normalize=True)
                z_assoc = z_assoc.squeeze(0)

            # ── Pillar 4: Neuromodulators (Noradrenaline, Dopamine, Acetylcholine)
            surprise_delta = total_surprise - self.surprise_ema
            self.surprise_ema = 0.90 * self.surprise_ema + 0.10 * total_surprise

            # Noradrenaline: sharp spike on context shift / surprise -> boosts plasticity & resets fatigue!
            na = 0.25 + 2.15 * (1.0 / (1.0 + math.exp(-5.0 * (surprise_delta + 0.05))))
            self.neuromod.noradrenaline = max(0.10, min(2.50, na))
            if self.neuromod.noradrenaline > 1.55:
                # Sudden context switch resets attention fatigue immediately
                self.fatigue_l1.mul_(0.35)
                self.fatigue_l2.mul_(0.35)
                self.fatigue_l3.mul_(0.35)
                self.fatigue_assoc.mul_(0.35)

            # Dopamine: intrinsic curiosity + learning progress + external valence
            da = 0.40 + 1.10 * intrinsic_curiosity + 0.60 * float(external_valence)
            self.neuromod.dopamine = max(-0.50, min(2.00, da))
            self.m4_values[best_m4_idx] = 0.85 * self.m4_values[best_m4_idx] + 0.15 * self.neuromod.dopamine

            # Acetylcholine: high on novel/surprising external input (M2 encode), lower when familiar/spontaneous (recall)
            ach_target = 0.88 if (not spontaneous and total_surprise > 0.18) else (0.30 if spontaneous else 0.45)
            self.neuromod.acetylcholine = max(0.05, min(0.98, 0.65 * self.neuromod.acetylcholine + 0.35 * ach_target))

            # ── Pillar 4: 3-Factor Neuromodulated Oja-Hebb Plasticity (ZERO Backprop!)
            lr_eff = 0.018 * (0.60 + 0.55 * self.neuromod.noradrenaline) * max(0.15, 0.70 + 0.40 * self.neuromod.dopamine)
            if spontaneous:
                lr_eff *= 0.35  # gentle background synaptic maintenance

            # Local Oja-Hebb update: dW = lr * (y x^T - alpha * y^2 * W) on sparse active rows only!
            self._oja_hebb_update(self.W_in_L1, z_l1, x, lr_eff, decay=0.04)
            self._oja_hebb_update(self.W_12, z_l2, z_l1, lr_eff, decay=0.04)
            self._oja_hebb_update(self.W_23, z_l3, z_l2, lr_eff, decay=0.04)
            self._oja_hebb_update(self.W_3a, z_assoc, z_l3, lr_eff, decay=0.04)

            # Predictive top-down generative weights minimize local reconstruction error locally!
            self._local_predictive_update(self.G_10, err_x, z_l1, lr_eff * 1.15)
            self._local_predictive_update(self.G_21, z_l1 - pred_l1, z_l2, lr_eff * 1.10)
            self._local_predictive_update(self.G_32, z_l2 - pred_l2, z_l3, lr_eff * 1.10)

            # Dopamine-gated eligibility trace on recurrent associative matrix W_assoc_rec
            active_a = torch.nonzero(z_assoc > 0, as_tuple=False).squeeze(-1)
            if active_a.numel() > 0:
                za_sub = z_assoc[active_a]
                coact_sub = torch.outer(za_sub, za_sub).to(dtype=self.dtype)
                # Decay eligibility trace & accumulate local co-activation
                self.E_assoc_rec.mul_(0.88)
                self.E_assoc_rec[active_a.unsqueeze(1), active_a.unsqueeze(0)] += 0.25 * coact_sub
                da_scale = float(0.015 * self.neuromod.dopamine * (0.5 + 0.5 * self.neuromod.noradrenaline))
                if abs(da_scale) > 1e-6:
                    self.W_assoc_rec.add_(self.E_assoc_rec, alpha=da_scale)
                    self.W_assoc_rec.fill_diagonal_(0.0)
                    self.W_assoc_rec.clamp_(-1.0, 1.0)

            # Update M4 prototype toward active associative state (competitive local Hebb)
            p_upd = F.normalize(
                (1.0 - 0.12) * self.m4_prototypes[best_m4_idx].float() + 0.12 * z_assoc, dim=-1, eps=1e-6
            )
            self.m4_prototypes[best_m4_idx] = p_upd.to(dtype=self.dtype)

            # ── Pillar 2 & 4: High ACh + Surprise (or pinned identity) -> Write Episode to M2 (Hippocampus)
            m2_encoded = False
            if (not spontaneous) and (pinned or total_surprise >= 0.25 or self.neuromod.acetylcholine >= 0.65):
                self._encode_m2_episode(z_assoc, x, total_surprise, self.neuromod.dopamine, pinned=bool(pinned))
                m2_encoded = True

            # ── Pillar 5: Firing-Rate Fatigue & Long-Term Threshold Homeostasis
            fat_decay = 0.80 if spontaneous else 0.84
            fat_add = 0.12 if spontaneous else 0.20
            for st_name, z_new, fat_name, thr_name, avg_name in (
                ("state_l1", z_l1, "fatigue_l1", "thresh_l1", "avg_act_l1"),
                ("state_l2", z_l2, "fatigue_l2", "thresh_l2", "avg_act_l2"),
                ("state_l3", z_l3, "fatigue_l3", "thresh_l3", "avg_act_l3"),
                ("state_assoc", z_assoc, "fatigue_assoc", "thresh_assoc", "avg_act_assoc"),
            ):
                setattr(self, st_name, z_new)
                fired = (z_new > 0).float()
                fat = getattr(self, fat_name)
                fat.mul_(fat_decay).add_(fat_add * fired)
                fat.clamp_(0.0, 1.2)
                avg = getattr(self, avg_name)
                avg.mul_(0.95).add_(0.05 * fired)
                thr = getattr(self, thr_name)
                thr.add_(0.015 * (avg - 0.04))
                thr.clamp_(-0.35, 0.35)

            mean_fatigue = float(
                (self.fatigue_l1.mean() + self.fatigue_l2.mean() + self.fatigue_l3.mean() + self.fatigue_assoc.mean()).item()
                / 4.0
            )

            # ── Pillar 1 & 6: Structural Neurogenesis Trigger ────────────────
            if total_surprise >= 0.38 or uncertainty >= 0.62:
                self.high_surprise_streak += 1
            else:
                self.high_surprise_streak = max(0, self.high_surprise_streak - 1)

            saturated = float((self.avg_act_assoc > 0.075).float().mean().item()) > 0.25
            neurons_added = 0
            ng_events = 0
            if allow_growth and (self.high_surprise_streak >= 3 or saturated):
                self.high_surprise_streak = 0
                neurons_added = self.grow_structure(
                    add_l1=128, add_l2=96, add_l3=64, add_assoc=96, add_m4=32, seed_input=x
                )
                if neurons_added > 0:
                    ng_events = 1

            budget_res = self.enforce_memory_budget()

            tel = LivingStepTelemetry(
                surprise=total_surprise,
                surprise_l1=surprise_l1,
                surprise_l2=surprise_l2,
                surprise_l3=surprise_x,
                predictive_suppression_pct=suppression_pct,
                intrinsic_curiosity=intrinsic_curiosity,
                learning_progress=learning_progress,
                boredom_exploration=boredom_exploration,
                neurogenesis_events=ng_events,
                neurons_added=neurons_added,
                pruned_synapses=budget_res["pruned_synapses"],
                pruned_episodes=budget_res["pruned_episodes"],
                m2_encoded=m2_encoded,
                theta_phase=self.theta_phase,
                gamma_phase=self.gamma_phase,
                mean_fatigue=mean_fatigue,
                neuromodulators=self.neuromod.to_dict(),
            )
            self.last_telemetry = tel
            return tel

    @torch.no_grad()
    def _oja_hebb_update(
        self,
        W: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        lr: float,
        decay: float = 0.04,
    ) -> None:
        """
        Sparse Oja-Hebb local rule (ONLY updates active postsynaptic rows y > 0 -> ultra fast!):
            dW_i = lr * y_i * (x - decay * y_i * W_i)
        """
        active = torch.nonzero(y > 0, as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            return
        y_a = y[active].unsqueeze(1)  # [k, 1]
        w_sub = W.index_select(0, active).float()  # [k, in_dim]
        dw = lr * y_a * (x.unsqueeze(0) - decay * y_a * w_sub)
        w_new = w_sub + dw
        # Keep rows norm-bounded locally
        norms = w_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        w_new = w_new * torch.clamp(1.2 / norms, max=1.0)
        W[active] = w_new.to(dtype=W.dtype)

    @torch.no_grad()
    def _local_predictive_update(
        self,
        G: torch.Tensor,
        err: torch.Tensor,
        src_sdr: torch.Tensor,
        lr: float,
    ) -> None:
        """Local Predictive Coding update on top-down generative weights G (ONLY active presynaptic cols!)."""
        active = torch.nonzero(src_sdr > 0, as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            return
        s_a = src_sdr[active].unsqueeze(0)  # [1, k]
        g_sub = G.index_select(1, active).float()  # [out_dim, k]
        dg = lr * torch.outer(err, src_sdr[active])
        g_new = g_sub + dg
        norms = g_new.norm(dim=0, keepdim=True).clamp(min=1e-6)
        g_new = g_new * torch.clamp(1.2 / norms, max=1.0)
        G[:, active] = g_new.to(dtype=G.dtype)

    @torch.no_grad()
    def _encode_m2_episode(
        self,
        assoc_sdr: torch.Tensor,
        sensory_vec: torch.Tensor,
        surprise: float,
        valence: float,
        pinned: bool = False,
    ) -> None:
        a_row = F.normalize(assoc_sdr.float(), dim=-1, eps=1e-6).to(dtype=self.dtype).unsqueeze(0)
        s_row = F.normalize(sensory_vec.float(), dim=-1, eps=1e-6).to(dtype=self.dtype).unsqueeze(0)
        # Merge if nearly identical to an existing M2 episode (> 0.94 cosine)
        if self.m2_assoc_traces.shape[0] > 0:
            sims = self.m2_assoc_traces.float() @ a_row.float().squeeze(0)
            max_sim, idx = torch.max(sims, dim=0)
            if float(max_sim.item()) >= 0.94:
                i = int(idx.item())
                self.m2_strengths[i] = min(5.0, float(self.m2_strengths[i].item()) + (0.60 if pinned else 0.25))
                self.m2_last_used[i] = self.tick_count
                if pinned:
                    self.m2_pinned[i] = True
                return

        self.m2_assoc_traces = torch.cat([self.m2_assoc_traces, a_row], dim=0)
        self.m2_sensory_traces = torch.cat([self.m2_sensory_traces, s_row], dim=0)
        init_strength = (2.5 + surprise) if pinned else (1.0 + surprise)
        self.m2_strengths = torch.cat([self.m2_strengths, torch.tensor([init_strength], device=self.device)], dim=0)
        self.m2_surprises = torch.cat([self.m2_surprises, torch.tensor([surprise], device=self.device)], dim=0)
        self.m2_last_used = torch.cat([self.m2_last_used, torch.tensor([self.tick_count], device=self.device, dtype=torch.int64)], dim=0)
        self.m2_valences = torch.cat([self.m2_valences, torch.tensor([valence], device=self.device)], dim=0)
        self.m2_pinned = torch.cat([self.m2_pinned, torch.tensor([bool(pinned)], device=self.device, dtype=torch.bool)], dim=0)

    # ── Sleep Offline Replay (M2 -> M1) + SHY Homeostasis (Pillar 1 & 4) ─────
    @torch.no_grad()
    def sleep_consolidate_replay(self, replay_steps: int = 12) -> dict[str, Any]:
        """
        Offline Sleep Consolidation (low ACh):
          - Replays high-surprise, high-valence & pinned episodes from M2 (Hippocampus) back into M1 (L1/L2/L3/Assoc)
          - Applies Synaptic Homeostasis (SHY) pruning to clear weak synaptic noise
          - Enforces memory budget cleanly while preserving all `m2_pinned` core traces
        """
        with self.lock:
            self.neuromod.acetylcholine = 0.15  # Low ACh = offline consolidation mode!
            self.neuromod.noradrenaline = 0.20
            m = int(self.m2_sensory_traces.shape[0])
            replayed = 0
            if m > 0:
                n_rep = min(m, max(1, int(replay_steps)))
                priority = self.m2_strengths + 0.8 * self.m2_surprises + 0.4 * self.m2_valences + self.m2_pinned.float() * 3.0
                top_idx = torch.topk(priority, k=n_rep, largest=True, sorted=False).indices.tolist()
                for idx in top_idx:
                    s_vec = self.m2_sensory_traces[idx].float()
                    self.cognitive_step(sensory_vec=s_vec, external_valence=0.35, allow_growth=False, spontaneous=True)
                    replayed += 1
                # Gentle biological forgetting curve on unpinned M2 traces (pinned traces NEVER decay)
                self.m2_strengths = torch.where(self.m2_pinned, self.m2_strengths, self.m2_strengths * 0.95)
                self.total_replay_cycles += 1

            pruned_syn = self._prune_weak_synapses(eps=0.012, downscale=0.94)
            # Reset neuronal fatigue completely after sleep!
            self.fatigue_l1.zero_()
            self.fatigue_l2.zero_()
            self.fatigue_l3.zero_()
            self.fatigue_assoc.zero_()
            budget_res = self.enforce_memory_budget()
            return {
                "replayed_episodes": replayed,
                "pruned_synapses": pruned_syn + budget_res["pruned_synapses"],
                "pruned_episodes": budget_res["pruned_episodes"],
                "living_mb": self.living_memory_mb(),
            }

    # ── Pillar 5: Continuous Background Spontaneous Activity Clock ───────────
    def set_paused(self, paused: bool) -> None:
        """Pause/resume background spontaneous GPU ticks cleanly during heavy GPU training or sleep."""
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def start_spontaneous_clock(self, interval_sec: float = 0.65) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                if not self._pause_event.is_set():
                    try:
                        # Non-blocking acquire so spontaneous background NEVER stalls chat or training
                        if self.lock.acquire(blocking=False):
                            try:
                                if not self._pause_event.is_set():
                                    self.cognitive_step(sensory_vec=None, allow_growth=False, spontaneous=True)
                            finally:
                                self.lock.release()
                    except Exception:
                        pass
                self._stop_event.wait(interval_sec)

        self._bg_thread = threading.Thread(target=_loop, daemon=True, name="LivingBrainClock")
        self._bg_thread.start()

    def stop_spontaneous_clock(self) -> None:
        self._stop_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.5)

    def reset(self) -> None:
        with self.lock:
            self.n_l1 = 512
            self.n_l2 = 384
            self.n_l3 = 256
            self.n_assoc = 384
            self.n_m4 = 64
            self.tick_count = 0
            self.high_surprise_streak = 0
            self.low_surprise_streak = 0
            self.total_neurogenesis_events = 0
            self.total_neurons_born = 0
            self.total_neurons_pruned = 0
            self.total_synapses_pruned = 0
            self.total_episodes_pruned = 0
            self.total_replay_cycles = 0
            self.total_explorations = 0
            self.neuromod = NeuromodulatorState()
            self._init_tensors()
            self.enforce_memory_budget()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            hw = self.hardware_memory_stats()
            eff_budget = self.effective_budget_mb()
            used_mb = hw["living_mb"]
            headroom_mb = round(max(0.0, eff_budget - used_mb), 2)
            util_pct = round(min(100.0, (used_mb / max(MIN_BUDGET_MB, eff_budget)) * 100.0), 1)
            return {
                "target_device": self.target_device_mode,
                "cuda_available": torch.cuda.is_available(),
                "memory_budget_mb": round(self.memory_budget_mb, 1),
                "effective_budget_mb": eff_budget,
                "min_budget_mb": MIN_BUDGET_MB,
                "max_budget_mb": MAX_BUDGET_MB,
                "living_mb": used_mb,
                "headroom_mb": headroom_mb,
                "utilization_pct": util_pct,
                "budget_status": self.budget_status,
                "gpu_allocated_mb": hw["gpu_allocated_mb"],
                "gpu_total_mb": hw["gpu_total_mb"],
                "ram_rss_mb": hw["ram_rss_mb"],
                "ram_total_mb": hw["ram_total_mb"],
                "neurons": {
                    "l1": self.n_l1,
                    "l2": self.n_l2,
                    "l3": self.n_l3,
                    "assoc": self.n_assoc,
                    "m4": self.n_m4,
                    "total": self.total_neurons_count(),
                },
                "active_synapses": self.active_synapses_count(),
                "m2_episodes": int(self.m2_assoc_traces.shape[0]),
                "m2_pinned_episodes": int(self.m2_pinned.sum().item()),
                "tick_count": self.tick_count,
                "total_neurogenesis_events": self.total_neurogenesis_events,
                "total_neurons_born": self.total_neurons_born,
                "total_neurons_pruned": self.total_neurons_pruned,
                "total_synapses_pruned": self.total_synapses_pruned,
                "total_episodes_pruned": self.total_episodes_pruned,
                "total_replay_cycles": self.total_replay_cycles,
                "total_explorations": self.total_explorations,
                "neuromodulators": self.neuromod.to_dict(),
                "last_step": self.last_telemetry.to_dict(),
            }

    def export_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "target_device_mode": self.target_device_mode,
                "memory_budget_mb": self.memory_budget_mb,
                "n_l1": self.n_l1,
                "n_l2": self.n_l2,
                "n_l3": self.n_l3,
                "n_assoc": self.n_assoc,
                "n_m4": self.n_m4,
                "tick_count": self.tick_count,
                "total_neurogenesis_events": self.total_neurogenesis_events,
                "total_neurons_born": self.total_neurons_born,
                "total_neurons_pruned": self.total_neurons_pruned,
                "total_synapses_pruned": self.total_synapses_pruned,
                "total_episodes_pruned": self.total_episodes_pruned,
                "total_replay_cycles": self.total_replay_cycles,
                "total_explorations": self.total_explorations,
                "tensors": {
                    k: getattr(self, k).detach().cpu()
                    for k in (
                        "W_in_L1", "G_10", "W_12", "G_21", "W_23", "G_32",
                        "W_3a", "W_assoc_rec", "E_assoc_rec",
                        "state_l1", "state_l2", "state_l3", "state_assoc",
                        "fatigue_l1", "fatigue_l2", "fatigue_l3", "fatigue_assoc",
                        "thresh_l1", "thresh_l2", "thresh_l3", "thresh_assoc",
                        "avg_act_l1", "avg_act_l2", "avg_act_l3", "avg_act_assoc",
                        "m2_assoc_traces", "m2_sensory_traces",
                        "m2_strengths", "m2_surprises", "m2_last_used", "m2_valences", "m2_pinned",
                        "m4_prototypes", "m4_visits", "m4_err_ema", "m4_values",
                    )
                },
            }

    @torch.no_grad()
    def import_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or "tensors" not in state:
            return
        with self.lock:
            want_mode = str(state.get("target_device_mode", self.target_device_mode))
            self.target_device_mode = "gpu" if (want_mode == "gpu" and torch.cuda.is_available()) else "ram"
            self.device = torch.device("cuda" if self.target_device_mode == "gpu" else "cpu")
            self.dtype = (
                torch.bfloat16
                if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
                else torch.float32
            )
            self.memory_budget_mb = max(MIN_BUDGET_MB, min(MAX_BUDGET_MB, float(state.get("memory_budget_mb", self.memory_budget_mb))))
            self.n_l1 = int(state.get("n_l1", self.n_l1))
            self.n_l2 = int(state.get("n_l2", self.n_l2))
            self.n_l3 = int(state.get("n_l3", self.n_l3))
            self.n_assoc = int(state.get("n_assoc", self.n_assoc))
            self.n_m4 = int(state.get("n_m4", self.n_m4))
            self.tick_count = int(state.get("tick_count", 0))
            self.total_neurogenesis_events = int(state.get("total_neurogenesis_events", 0))
            self.total_neurons_born = int(state.get("total_neurons_born", 0))
            self.total_neurons_pruned = int(state.get("total_neurons_pruned", 0))
            self.total_synapses_pruned = int(state.get("total_synapses_pruned", 0))
            self.total_episodes_pruned = int(state.get("total_episodes_pruned", 0))
            self.total_replay_cycles = int(state.get("total_replay_cycles", 0))
            self.total_explorations = int(state.get("total_explorations", 0))

            mat_set = {
                "W_in_L1", "G_10", "W_12", "G_21", "W_23", "G_32",
                "W_3a", "W_assoc_rec", "E_assoc_rec",
                "m2_assoc_traces", "m2_sensory_traces", "m4_prototypes",
            }
            tensors = state["tensors"]
            for k, t in tensors.items():
                if isinstance(t, torch.Tensor):
                    if k in mat_set:
                        setattr(self, k, t.to(device=self.device, dtype=self.dtype).contiguous())
                    elif k == "m2_pinned":
                        setattr(self, k, t.to(device=self.device, dtype=torch.bool).contiguous())
                    else:
                        setattr(self, k, t.to(device=self.device).contiguous())
            m2_n = int(self.m2_assoc_traces.shape[0])
            if "m2_pinned" not in tensors or int(self.m2_pinned.shape[0]) != m2_n:
                self.m2_pinned = torch.zeros((m2_n,), device=self.device, dtype=torch.bool)
            self.enforce_memory_budget()
