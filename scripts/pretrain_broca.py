from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct invocation: `python scripts/pretrain_broca.py ...`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from brain.agent import SleepingBrain
from brain.config import BrainConfig
from brain.curriculum import ELEMENTARY_WEB_SYLLABUS
from brain.language_trainer import build_denoising_corpus, iter_utf8_text_chunks
from brain.web_learner import chunk_text, extract_lead_definition, gather_web_sources


def run_pretrain(
    data_path: str | None = None,
    base_dir: str | Path = ROOT_DIR,
    state_dir: str = "brain_state",
    epochs: int = 4,
    batch_size: int = 32,
    lr: float | None = None,
    early_stop_loss: float = 0.90,
    max_chunks: int | None = None,
    max_steps_per_epoch: int | None = None,
    web_topics: int = 0,
    device_override: str | None = None,
) -> dict[str, object]:
    """
    Pre-train WernickeByteCortex + BrocaByteTransformer in bfloat16 via denoising autoencoding
    through the k-WTA SDR bottleneck:
      - If `data_path` is provided: streams UTF-8 `.txt` file(s) in 1 MB chunks (safe for 100-300 MB dumps).
      - Otherwise: uses `brain/curriculum.py`, `BASE_HUMAN_DIALOGUES`, and optional `web_learner.py` articles.
    """
    cfg = BrainConfig(state_dir=state_dir, language_bootstrap_epochs=0)
    brain = SleepingBrain(cfg=cfg, base_dir=base_dir, load_language_mode="none")
    if device_override:
        dev = torch.device(device_override)
        brain.device = dev
        brain.language_cortex.device = dev
        brain.language_cortex.wernicke_module.to(dev)
        brain.language_cortex.broca_module.to(dev)
    brain.language_cortex.load_sync()

    dev_name = (
        torch.cuda.get_device_name(0)
        if brain.language_cortex.device.type == "cuda"
        else "CPU"
    )
    print(f"[pretrain_broca] Urządzenie: {dev_name} | bfloat16 autocast: {brain.language_cortex.device.type == 'cuda'}")
    print(
        f"[pretrain_broca] Parametry Wernicke+Broca: "
        f"{brain.language_cortex.total_parameters() / 1e6:.2f}M | "
        f"SDR k-WTA: {cfg.sdr_k}/{cfg.num_neurons} ({100.0 * cfg.sdr_k / cfg.num_neurons:.2f}%)"
    )

    if data_path:
        src = Path(data_path)
        if src.suffix.lower() == ".parquet":
            from brain.language_trainer import iter_parquet_polish_pairs

            steps = max_steps_per_epoch or max(50, epochs * 500)
            print(f"[pretrain_broca] Tryb strumieniowy Parquet (polski + gramatyka) z: {src} ({steps} kroków)")
            stats = brain.language_cortex.train_parquet_polish_stream(
                pair_iterable_factory=lambda: iter_parquet_polish_pairs(
                    src,
                    continuation_ratio=0.35,
                    max_pairs=max_chunks,
                ),
                max_steps=steps,
                batch_size=batch_size,
                lr=lr,
                early_stop_loss=early_stop_loss,
                mask_prob=0.16,
            )
            stats["epochs_run"] = 1
        else:
            print(f"[pretrain_broca] Tryb strumieniowy UTF-8 z: {src}")
            stats = brain.language_cortex.train_streaming_batches(
                chunk_iterable_factory=lambda: iter_utf8_text_chunks(
                    src,
                    chunk_chars=220,
                    min_chars=20,
                    max_chunks=max_chunks,
                ),
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                early_stop_loss=early_stop_loss,
                mask_prob=0.18,
                max_steps_per_epoch=max_steps_per_epoch,
            )
    else:
        web_texts: list[str] = []
        if web_topics > 0:
            seen_urls: set[str] = set()
            for topic in ELEMENTARY_WEB_SYLLABUS[:web_topics]:
                docs = gather_web_sources(topic, seen_urls=seen_urls, max_docs=1, timeout=10, max_workers=2)
                for doc in docs:
                    lead = extract_lead_definition(doc)
                    if lead:
                        web_texts.append(lead)
                    for ch in chunk_text(doc.text, focus_query=topic, max_chars=2000, max_chunks=1):
                        web_texts.extend(brain.language_cortex.extract_sensory_stimuli(ch, max_items=6))
        corpus = build_denoising_corpus(
            extra_texts=web_texts,
            include_base=True,
            include_curriculum=True,
            max_samples=max_chunks or 1200,
        )
        print(f"[pretrain_broca] Tryb domyślny (curriculum + BASE_HUMAN_DIALOGUES + sieć={len(web_texts)}): {len(corpus)} próbek")
        stats = brain.language_cortex.train_streaming_batches(
            chunk_iterable_factory=lambda: iter(corpus),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            early_stop_loss=early_stop_loss,
            mask_prob=0.18,
            max_steps_per_epoch=max_steps_per_epoch,
        )

    ckpt_path = brain.save_state()
    print(
        f"[pretrain_broca] GOTOWE | epok={stats['epochs_run']} | próbek={stats['samples']} | "
        f"loss: {stats['loss_start']} -> {stats['loss_end']} | płynność: {stats['fluency']:.1f}% | "
        f"czas: {stats['duration_ms']:.0f} ms | zapisano: {ckpt_path}"
    )
    return {**stats, "checkpoint": str(ckpt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-training BrocaByteTransformer + WernickeByteCortex (denoising autoencoding przez SDR k-WTA w bfloat16)"
    )
    parser.add_argument("--data", type=str, default=None, help="Ścieżka do pliku lub folderu .txt UTF-8 (strumieniowy loader)")
    parser.add_argument("--base-dir", type=str, default=str(ROOT_DIR), help="Katalog bazowy projektu")
    parser.add_argument("--state-dir", type=str, default="brain_state", help="Podkatalog checkpointów")
    parser.add_argument("--epochs", type=int, default=6, help="Liczba epok treningu")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate AdamW (opcjonalnie)")
    parser.add_argument("--early-stop-loss", type=float, default=0.90, help="Próg early stoppingu (domyślnie 0.90)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunków na epokę (opcjonalnie)")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit batchy na epokę (opcjonalnie)")
    parser.add_argument("--web-topics", type=int, default=0, help="Ile tematów dociągnąć z web_learner.py w trybie domyślnym")
    parser.add_argument("--device", type=str, default=None, help="Wymuś urządzenie (cuda / cpu)")
    args = parser.parse_args()

    run_pretrain(
        data_path=args.data,
        base_dir=args.base_dir,
        state_dir=args.state_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stop_loss=args.early_stop_loss,
        max_chunks=args.max_chunks,
        max_steps_per_epoch=args.max_steps,
        web_topics=args.web_topics,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
