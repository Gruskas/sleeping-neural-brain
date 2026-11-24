from dataclasses import dataclass


@dataclass
class BrainConfig:
    # Associative Concept Cortex (expanded capacity)
    max_concepts: int = 16384
    max_relations: int = 512
    concept_dim: int = 64
    relation_dim: int = 32

    # Energy / Sleep Pressure Homeostasis (pure chat + study driven)
    max_energy: float = 100.0
    teach_energy_drain: float = 2.0
    chat_energy_drain: float = 3.2
    study_page_energy_drain: float = 4.5
    drowsy_threshold: float = 20.0
    sleep_threshold: float = 0.0

    # Expanded Memory Limits (Short-Term Hippocampus & Long-Term Cortex)
    max_short_term_memories: int = 5000
    max_short_term_facts: int = 5000
    max_semantic_memories: int = 50000
    max_rem_reflections: int = 500
    working_chat_window: int = 24
    working_chat_buffer: int = 48
    chat_log_limit: int = 500

    # Retrieval depths during conversation
    recall_top_hippo: int = 14
    recall_top_semantic: int = 16
    recall_top_rem: int = 8

    # Sleep Consolidation (NREM + REM) on GPU
    nrem_concept_epochs: int = 120
    concept_lr: float = 5e-3
    ewc_lambda_concept: float = 8.0
    rem_similarity_threshold: float = 0.56
    rem_max_new_links_per_night: int = 20

    # Self-Contained PyTorch Neural Language & Semantic Cortex (100% local on GPU)
    language_model_id: str = "Autonomiczna-Kora-PyTorch-v2"
    vocab_size: int = 16384
    ngram_buckets: int = 2048
    embed_dim: int = 256
    cortex_dim: int = 384
    cortex_layers: int = 6
    cortex_heads: int = 8
    cortex_ff_dim: int = 1536
    cortex_max_seq_len: int = 320
    cortex_sleep_epochs: int = 14
    language_bootstrap_epochs: int = 8
    language_train_epochs: int = 22
    cortex_lr: float = 9e-4
    lora_max_replay_samples: int = 80
    embed_batch_size: int = 32
    max_new_tokens: int = 2400
    memory_decay_per_night: float = 0.96
    memory_prune_threshold: float = 0.18

    # Autonomous Web Study settings
    study_max_rounds: int = 4
    study_pages_per_round: int = 3
    study_target_mastery: float = 85.0
    study_http_timeout: int = 12
    study_fetch_workers: int = 3

    # Persistence
    state_dir: str = "brain_state"
