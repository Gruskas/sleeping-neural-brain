from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.corpus_fetcher import (
    CorpusPaths,
    CorpusSanitizer,
    build_2hop_knowledge_graph,
    is_clean_polish_utf8,
    load_corpus_2hop_facts,
    load_corpus_qa_pairs,
    sanitize_polish_text,
)
from brain.hippocampus import EngramMeta, Hippocampus
from brain.language_trainer import iter_processed_corpus_pairs


class TestCorpusFetcher(unittest.TestCase):
    def test_01_utf8_sanitizer_and_deduplication(self) -> None:
        raw_wiki = "== Historia ==\nPolska leży w Europie Środkowej[1]. https://example.com <b>Test</b>"
        clean = sanitize_polish_text(raw_wiki)
        self.assertNotIn("https://", clean)
        self.assertNotIn("<b>", clean)
        self.assertNotIn("[1]", clean)
        self.assertTrue(is_clean_polish_utf8(clean, min_chars=20))

        # Reject excessive digits / repetition / non-Latin noise
        self.assertFalse(is_clean_polish_utf8("1234567890 1234567890 1234567890 123456", min_chars=15))
        self.assertFalse(is_clean_polish_utf8("aaaaaaaaaaaa bbbbbbbbbbbb cccccccccccc", min_chars=15))

        san = CorpusSanitizer(min_chars=20, max_utf8_bytes=600)
        r1 = san.accept("Kraków był historyczną stolicą Polski przez wiele stuleci.")
        r2 = san.accept("  kraków był historyczną stolicą polski przez wiele stuleci. ")
        self.assertIsNotNone(r1)
        self.assertIsNone(r2)
        self.assertEqual(san.duplicates, 1)

    def test_02_2hop_graph_builder_and_hippocampus_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = CorpusPaths.from_base_dir(tmp)
            # Sample edges with A -> B -> C structure (no direct A -> C edge)
            cn_edges = [
                {"source": "conceptnet_pl", "a": "kot", "rel": "IsA", "b": "ssak", "weight": 2.0},
                {"source": "conceptnet_pl", "a": "ssak", "rel": "CapableOf", "b": "oddychanie płucami", "weight": 2.0},
                {"source": "conceptnet_pl", "a": "słońce", "rel": "Causes", "b": "parowanie", "weight": 1.5},
                {"source": "conceptnet_pl", "a": "parowanie", "rel": "Causes", "b": "chmura", "weight": 1.5},
            ]
            wn_edges = [
                {"source": "plwordnet", "a": "drzewo", "rel": "hypernym", "b": "roślina", "weight": 1.0},
                {"source": "plwordnet", "a": "roślina", "rel": "meronym", "b": "liść", "weight": 1.0},
            ]
            (paths.processed / "conceptnet_pl_edges.jsonl").write_text(
                "\n".join(json.dumps(e, ensure_ascii=False) for e in cn_edges) + "\n", encoding="utf-8"
            )
            (paths.processed / "plwordnet_edges.jsonl").write_text(
                "\n".join(json.dumps(e, ensure_ascii=False) for e in wn_edges) + "\n", encoding="utf-8"
            )
            (paths.processed / "tinystories_pl.jsonl").write_text(
                json.dumps({"source": "tinystories_pl", "text": "Mały piesek pobiegł na zieloną łąkę i znalazł czerwoną piłkę."}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (paths.processed / "polqa.jsonl").write_text(
                json.dumps({"source": "polqa", "q": "Gdzie leży Kraków?", "a": "nad Wisłą", "passage": "Kraków leży w południowej Polsce nad rzeką Wisłą."}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            kg = build_2hop_knowledge_graph(paths, max_facts=50, max_2hop=50, max_negatives=20)
            self.assertEqual(kg["facts"], 6)
            self.assertGreaterEqual(kg["paths_2hop"], 3)

            twohop_rows = [json.loads(ln) for ln in (paths.processed / "knowledge_graph_2hop.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertTrue(any(r["a"] == "kot" and r["b"] == "ssak" and r["c"] == "oddychanie płucami" for r in twohop_rows))

            pairs = list(iter_processed_corpus_pairs(tmp, min_chars=20, max_pairs=10))
            self.assertGreaterEqual(len(pairs), 1)
            qa = load_corpus_qa_pairs(tmp, max_pairs=20)
            self.assertTrue(any("Kraków" in q for q, _ in qa))
            facts = load_corpus_2hop_facts(tmp, max_items=50)
            self.assertGreaterEqual(len(facts), 6)

            # Verify loading into Hippocampus engrams works cleanly
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hippo = Hippocampus(num_neurons=512, sdr_k=32, device=dev)
            sdrs = torch.randn((3, 512), device=dev)
            metas = [EngramMeta(preview_label=facts[i]["text"], topic=facts[i]["topic"]) for i in range(3)]
            hippo.semantic.add_batch(sdrs, metas, [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], day=1)
            self.assertEqual(len(hippo.semantic), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
