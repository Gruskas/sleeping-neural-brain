from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from .config import BrainConfig
from .hippocampus import Hippocampus
from .neural_cortex import BiologicalAssociativeCortex
from .sensory_wernicke import k_wta


def _locked(lock: threading.RLock | None):
    return lock if lock is not None else contextlib.nullcontext()


@dataclass
class SleepReport:
    night_number: int
    duration_ms: float
    device_name: str
    # Stage 1: NREM Light
    new_memories: int = 0
    updated_memories: int = 0  # merged into generalized SDR prototypes
    forgotten_memories: int = 0
    new_facts: int = 0
    reinforced_facts: int = 0
    # Stage 2: NREM Deep (Local Predictive Coding + Hebb + SHY + optional Broca)
    prediction_error_start: float = 0.0
    prediction_error_end: float = 0.0
    sdr_sparsity_pct: float = 4.0
    pruned_synapses: int = 0
    active_synapses: int = 0
    attractor_steps: int = 4
    broca_loss_start: float = 0.0
    broca_loss_end: float = 0.0
    # Backwards-compatible UI/API aliases
    concept_loss_start: float = 0.0
    concept_loss_end: float = 0.0
    lora_trained: bool = False
    lora_samples: int = 0
    lora_loss_start: float = 0.0
    lora_loss_end: float = 0.0
    lora_duration_ms: float = 0.0
    # Stage 3: Vectorized GPU REM Dreaming
    rem_new_links: list[dict[str, Any]] = field(default_factory=list)
    rem_reflections: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SleepEngine:
    """
    Fully Vectorized GPU Sleep Engine (zero nested Python loops, zero string templates):
      1) NREM Light  : GPU engram similarity matmul, prototype merging (no duplicates), forgetting curve.
      2) NREM Deep   : Experience Replay (~30% fresh + ~70% long-term) -> 100% local cortical plasticity
                       (@torch.no_grad() Predictive Coding + Hebb + threshold homeostasis) ->
                       Synaptic Homeostasis Hypothesis (SHY: W_rec *= 0.94 + weak weight pruning) ->
                       optional short Broca denoising autoencoder refinement with early stopping.
      3) REM Dreaming: Vectorized GPU sampling of distant long-term engram pairs -> random neuron mask
                       + Gaussian noise -> parallel Modern Hopfield + W_rec attractor settling ->
                       store hybrid REM engrams -> decode short thought labels via BrocaByteTransformer.
    """

    def __init__(
        self,
        cfg: BrainConfig,
        cortex: BiologicalAssociativeCortex,
        hippocampus: Hippocampus,
        device: torch.device,
        language_cortex: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.cortex = cortex
        self.hippocampus = hippocampus
        self.device = device
        self.language_cortex = language_cortex

    def run_sleep_cycle(
        self,
        night_number: int,
        stage_callback: Callable[[str], None] | None = None,
        visual_pacing: bool = False,
        state_lock: threading.RLock | None = None,
        wake_texts_for_broca: list[str] | None = None,
    ) -> SleepReport:
        t0 = time.perf_counter()
        dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        report = SleepReport(
            night_number=night_number,
            duration_ms=0.0,
            device_name=dev_name,
            sdr_sparsity_pct=round(100.0 * (self.cfg.sdr_k / float(self.cfg.num_neurons)), 2),
            attractor_steps=self.cfg.attractor_steps,
        )

        # ── STAGE 1: NREM LIGHT (GPU prototype merging + forgetting) ─────────
        if stage_callback:
            stage_callback("NREM_LIGHT")

        with _locked(state_lock):
            light_res = self.hippocampus.consolidate_nrem_light_gpu(
                merge_threshold=self.cfg.engram_merge_threshold,
                decay=self.cfg.memory_decay_per_night,
                prune_threshold=self.cfg.memory_prune_threshold,
                current_day=night_number,
            )
            report.new_memories = int(light_res["new_memories"])
            report.updated_memories = int(light_res["merged_memories"])
            report.reinforced_facts = int(light_res["merged_memories"])
            report.new_facts = int(light_res["new_memories"])
            report.forgotten_memories = int(light_res["forgotten_memories"])
            fresh_engrams: torch.Tensor = light_res["fresh_engrams"]

        if visual_pacing:
            time.sleep(0.20)

        # ── STAGE 2: NREM DEEP (Experience Replay + Local Plasticity + SHY) ──
        if stage_callback:
            stage_callback("NREM_DEEP")

        with _locked(state_lock):
            replay_batch = self.hippocampus.sample_nrem_replay_batch(
                fresh_engrams=fresh_engrams,
                batch_size=self.cfg.nrem_replay_batch_size,
                fresh_ratio=0.30,
            )
            if replay_batch.shape[0] > 0:
                if self.language_cortex is not None:
                    with torch.no_grad():
                        target_cols = self.language_cortex.wernicke.sdr_to_columns(
                            replay_batch.to(device=self.device, dtype=torch.float32)
                        )
                else:
                    target_cols = (replay_batch @ self.cortex.W_in).view(
                        replay_batch.shape[0], self.cfg.num_columns, self.cfg.column_dim
                    )
                pc_stats = self.cortex.learn_local(
                    sensory_columns=target_cols,
                    z_batch=replay_batch,
                    steps=self.cfg.nrem_sleep_steps,
                    noise_std=0.04,
                )
                report.prediction_error_start = float(pc_stats["prediction_error_start"])
                report.prediction_error_end = float(pc_stats["prediction_error_end"])
                report.concept_loss_start = report.prediction_error_start
                report.concept_loss_end = report.prediction_error_end

            # Synaptic Homeostasis Hypothesis (SHY): W_rec *= 0.94 + weak synapse pruning
            shy_stats = self.cortex.apply_synaptic_homeostasis(
                downscale=self.cfg.shy_downscale,
                prune_eps=self.cfg.shy_prune_eps,
            )
            report.pruned_synapses = int(shy_stats["pruned_synapses"])
            report.active_synapses = int(shy_stats["active_synapses"])

        # Optional short Broca denoising autoencoder refinement (outside state_lock)
        if self.language_cortex is not None and self.language_cortex.ready:
            broca_texts = [t for t in (wake_texts_for_broca or []) if t and len(t.strip()) >= 8]
            if broca_texts or getattr(self.language_cortex, "_learned_qa_replay", None):
                b_stats = self.language_cortex.train_denoising_sleep(
                    texts=broca_texts,
                    epochs=self.cfg.cortex_sleep_epochs,
                    cortex=self.cortex,
                )
                report.lora_trained = bool(b_stats.get("trained", False))
                report.lora_samples = int(b_stats.get("samples", 0))
                report.broca_loss_start = float(b_stats.get("loss_start", 0.0))
                report.broca_loss_end = float(b_stats.get("loss_end", 0.0))
                report.lora_loss_start = report.broca_loss_start
                report.lora_loss_end = report.broca_loss_end
                report.lora_duration_ms = float(b_stats.get("duration_ms", 0.0))

        if visual_pacing:
            time.sleep(0.20)

        # ── STAGE 3: VECTORIZED GPU REM DREAMING ─────────────────────────────
        if stage_callback:
            stage_callback("REM_DREAMING")

        rem_cols: torch.Tensor | None = None
        rem_sdrs: torch.Tensor | None = None
        rem_pairs_meta: list[dict[str, Any]] = []

        with _locked(state_lock):
            sem_n = len(self.hippocampus.semantic)
            if sem_n >= 2:
                rem_sdrs, rem_cols, rem_pairs_meta = self._run_vectorized_rem_gpu()

        decoded_rem_labels: list[str] = []
        if rem_sdrs is not None and rem_cols is not None and rem_sdrs.shape[0] > 0:
            if self.language_cortex is not None and self.language_cortex.ready:
                for i in range(rem_cols.shape[0]):
                    gen = self.language_cortex.decode_columns(
                        rem_cols[i : i + 1],
                        max_new_bytes=72,
                        temperature=min(0.92, max(0.78, self.cfg.broca_temperature + 0.05)),
                    )
                    decoded_rem_labels.append(gen["text"] or "REM-attractor")
            else:
                decoded_rem_labels = [f"REM-hybrid-{night_number}-{i + 1}" for i in range(rem_sdrs.shape[0])]

            with _locked(state_lock):
                rem_ids = self.hippocampus.record_rem_engrams(
                    sdrs=rem_sdrs,
                    decoded_labels=decoded_rem_labels,
                    day=night_number,
                    sparsity_pct=report.sdr_sparsity_pct,
                )
                for idx_r, r_id in enumerate(rem_ids):
                    if idx_r < len(rem_pairs_meta):
                        rem_pairs_meta[idx_r]["rem_id"] = r_id

            report.rem_reflections = decoded_rem_labels
            report.rem_new_links = rem_pairs_meta

        with _locked(state_lock):
            self.hippocampus.clear_short_term()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return report

    @torch.no_grad()
    def _run_vectorized_rem_gpu(self) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """
        Fully vectorized GPU REM dreaming:
          1. Sample distant engram pairs from Pamięć trwała on GPU (low cosine similarity).
          2. Blend each pair with random binary neuron mask + Gaussian noise.
          3. Run parallel attractor settling (Modern Hopfield + W_rec + k-WTA) to novel hybrid z_rem.
        """
        sem = self.hippocampus.semantic.engrams  # [M, N] QuantizedSynapseMatrix (FP8)
        m = int(sem.shape[0])
        pool_n = min(m, 128)
        if pool_n < m:
            pool_idx = torch.randperm(m, device=self.device)[:pool_n]
            pool = sem.dequant_rows(pool_idx, out_dtype=torch.float32) if hasattr(sem, "dequant_rows") else sem.index_select(0, pool_idx).float()
        else:
            pool_idx = torch.arange(m, device=self.device)
            pool = sem.float()

        # Pairwise cosine similarity on GPU: pick the most distant pairs for creative synthesis
        sim = pool @ pool.T  # [P, P]
        eye = torch.eye(pool_n, device=self.device, dtype=torch.bool)
        # Mask diagonal and lower triangle so each unordered pair appears once
        upper_mask = torch.triu(torch.ones((pool_n, pool_n), dtype=torch.bool, device=self.device), diagonal=1)
        masked_sim = sim.masked_fill(~upper_mask | eye, 2.0)

        b_rem = min(int(self.cfg.rem_batch_size), max(1, (pool_n * (pool_n - 1)) // 2))
        flat_sim = masked_sim.reshape(-1)
        # Smallest similarities = most distant engrams
        pair_vals, flat_idx = torch.topk(flat_sim, k=b_rem, largest=False)
        row_i = flat_idx // pool_n
        col_j = flat_idx % pool_n

        e_a = pool.index_select(0, row_i)  # [B_rem, N]
        e_b = pool.index_select(0, col_j)  # [B_rem, N]

        # Random neuron masking + Gaussian noise
        mask = (torch.rand_like(e_a) > 0.5).float()
        noise = torch.randn_like(e_a) * float(self.cfg.rem_noise_std)
        mixed_raw = mask * e_a + (1.0 - mask) * e_b + noise
        z_seed, _, _ = k_wta(mixed_raw, k=self.cfg.sdr_k, normalize=True)

        sdr_to_col = self.language_cortex.wernicke.sdr_to_columns if self.language_cortex is not None else None
        if sdr_to_col is not None:
            s_seed = sdr_to_col(z_seed)
        else:
            s_seed = (z_seed @ self.cortex.W_in).view(b_rem, self.cfg.num_columns, self.cfg.column_dim)

        thought = self.cortex.perceive_and_think(
            sensory_columns=s_seed,
            sensory_sdr=z_seed,
            engram_bank=sem,
            sdr_to_columns_fn=sdr_to_col,
            steps=self.cfg.attractor_steps,
            noise_std=float(self.cfg.rem_noise_std) * 0.5,
            update_last_active=False,
            force_hopfield=True,
        )

        # Record purely numeric/structural REM link telemetry (no string templates)
        links: list[dict[str, Any]] = []
        src_ids = pool_idx.index_select(0, row_i).cpu().tolist()
        dst_ids = pool_idx.index_select(0, col_j).cpu().tolist()
        sims_cpu = pair_vals.cpu().tolist()
        for a_i, b_i, s_val in zip(src_ids, dst_ids, sims_cpu):
            links.append(
                {
                    "src_engram": self.hippocampus.semantic.meta[a_i].id,
                    "dst_engram": self.hippocampus.semantic.meta[b_i].id,
                    "pair_similarity": round(float(s_val), 4),
                    "hopfield_energy": thought.hopfield_energy,
                    "source": "REM_DREAM",
                }
            )

        return thought.z_star.to(torch.bfloat16), thought.columns_star, links
