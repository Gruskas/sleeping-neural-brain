from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrainConfig:
    # ── Tokenless UTF-8 Byte Architecture (Wernicke -> Hopfield Cortex -> Broca) ──
    language_model_id: str = "Wernicke-Hopfield-Broca-UTF8"
    architecture_label: str = "Bajty UTF-8 · Modern Hopfield · SDR k-WTA"

    # Raw UTF-8 bytes (0..255) + 4 control bytes (<PAD>=256, <BOS>=257, <EOS>=258, <MASK>=259)
    byte_vocab_size: int = 260
    patch_size: int = 4
    patch_stride: int = 4
    byte_embed_dim: int = 64
    max_input_bytes: int = 4096

    # Shared Cortical Columns S, Z* in R^{16 x D} + Sequential Patch Columns [B, P_active, D]
    num_columns: int = 16
    column_dim: int = 768
    wernicke_layers: int = 6
    wernicke_heads: int = 12
    wernicke_ff_dim: int = 2304
    wernicke_max_patch_slots: int = 64
    wernicke_rf_anchor_weight: float = 0.12

    # Biological Associative Cortex (Modern Continuous Hopfield + k-WTA SDR + Local Plasticity)
    bio_synapse_dtype: str = "fp8"  # Native Blackwell FP8 E4M3 per-row (fallback: int8 / bf16)
    num_neurons: int = 12288
    sdr_k: int = 492  # ~4.00% sparsity (within 3-5% biological k-WTA range)
    hopfield_beta: float = 12.0
    attractor_steps: int = 4  # 3-6 iterations
    cortex_rec_weight: float = 0.25
    cortex_pc_col_weight: float = 0.20
    cortex_hopfield_weight: float = 0.35
    pc_lr: float = 0.035
    hebb_lr: float = 0.018
    hebb_decay: float = 0.004
    homeostasis_lr: float = 0.02
    shy_downscale: float = 0.94
    shy_prune_eps: float = 0.008
    nrem_sleep_steps: int = 20
    nrem_replay_batch_size: int = 48
    rem_batch_size: int = 6
    rem_noise_std: float = 0.12
    engram_merge_threshold: float = 0.92

    # Broca Byte-Patching Transformer Decoder (Cross-Attention ONLY on Z*, head_dim=64 Tensor-Core optimal)
    broca_dim: int = 768
    broca_layers: int = 10
    broca_heads: int = 12
    broca_ff_dim: int = 2304
    # Sparse Mixture-of-Experts (MoE) in Broca decoder FFN:
    # top_k=2 * 1152 = 2304 matches dense FFN active params; 8 experts -> 4x FFN capacity (1152 = 9*128 Tensor-Core optimal)
    broca_use_moe: bool = True
    broca_num_experts: int = 8
    broca_moe_top_k: int = 2
    broca_moe_expert_dim: int = 1152
    broca_moe_aux_loss_coef: float = 0.01
    broca_moe_z_loss_coef: float = 0.001
    broca_dropout: float = 0.2
    broca_max_seq_len: int = 1024
    broca_lr: float = 1.2e-3
    broca_weight_decay: float = 0.05
    broca_label_smoothing: float = 0.1
    broca_dialogue_lr: float = 9e-4
    broca_dialogue_label_smoothing: float = 0.05
    broca_dialogue_early_stop_loss: float = 0.53
    broca_decoder_byte_dropout: float = 0.15
    broca_early_stop_loss: float = 0.90
    broca_z_noise_std: float = 0.03
    broca_z_dropout: float = 0.05
    broca_temperature: float = 0.55
    broca_top_p: float = 0.85
    broca_repetition_penalty: float = 1.18
    broca_max_new_bytes: int = 1024

    # Training epochs & GPU optimization (RTX 5080 / CUDA 12.8)
    language_bootstrap_epochs: int = 5
    language_train_epochs: int = 180
    cortex_sleep_epochs: int = 2
    broca_batch_size: int = 24
    embed_batch_size: int = 32
    enable_torch_compile: bool = False
    broca_gradient_checkpointing: bool = True

    # Informational Sleep & Creativity (no artificial fatigue)
    max_energy: float = 100.0
    auto_sleep_pending_threshold: int = 24
    creativity_level: float = 0.80

    # Autonomous 2-hop Concept Graph (DMN idle thinking & REM bridge synthesis)
    dmn_enabled: bool = True
    idle_think_delay_sec: float = 50.0
    dmn_think_interval_sec: float = 40.0
    rem_2hop_steps: int = 5
    dmn_bridge_min_hop_sim: float = 0.45
    dmn_bridge_max_direct_sim: float = 0.25

    # Memory Limits (Expanded Prefrontal Working Memory + Tensor Engram Banks on GPU)
    max_short_term_memories: int = 50000
    max_semantic_memories: int = 200000
    max_rem_reflections: int = 2000
    working_sdr_capacity: int = 64
    working_memory_slots: int = 8
    working_chat_window: int = 128
    working_chat_buffer: int = 256
    chat_log_limit: int = 2000
    self_critique_candidates: int = 3
    system2_reasoning_steps: int = 3
    self_correction_enabled: bool = True
    self_correction_threshold: float = 0.78

    # Recall top-k depths (tensor similarity)
    recall_top_hippo: int = 16
    recall_top_semantic: int = 24
    recall_top_rem: int = 8

    # Biological forgetting curve
    memory_decay_per_night: float = 0.96
    memory_prune_threshold: float = 0.18

    # Autonomous Web Study settings
    study_max_rounds: int = 4
    study_pages_per_round: int = 3
    study_target_mastery: float = 85.0
    study_http_timeout: int = 12
    study_fetch_workers: int = 3

    # Optional Teacher / Hybrid LLM (Default OFF in chat -> 100% self-sufficient brain; Bielik used on-demand as Teacher)
    use_local_llm_broca: bool = False
    teacher_model_id: str = "MassivDash/Bielik-4.5B-v3-heretic"
    broca_llm_model_id: str = "MassivDash/Bielik-4.5B-v3-heretic"
    broca_llm_fallback_model_id: str = "MassivDash/Bielik-4.5B-v3-heretic"
    broca_llm_max_new_tokens: int = 2048
    broca_llm_lora_r: int = 16
    broca_llm_lora_alpha: int = 32
    broca_llm_lora_lr: float = 1.5e-4
    broca_llm_sleep_epochs: int = 2

    # Persistence
    state_dir: str = "brain_state"
