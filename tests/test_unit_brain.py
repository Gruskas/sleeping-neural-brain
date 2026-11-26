from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from brain.agent import CHECKPOINT_VERSION, SleepingBrain
from brain.config import BrainConfig
from brain.hippocampus import Hippocampus
from brain.language_cortex import LanguageCortex
from brain.language_trainer import iter_utf8_text_chunks
from brain.neural_cortex import BiologicalAssociativeCortex
from brain.neural_language import BrocaByteTransformer
from brain.sensory_wernicke import WernickeByteCortex, encode_utf8_bytes
from brain.sleep_engine import SleepEngine
from brain.web_learner import (
    WebDocument,
    clean_search_query,
    extract_lead_definition,
    gather_web_sources,
    is_relevant_doc,
)
from server import BrainServer


def _start_test_web_server():
    class WebHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/article.html":
                html_doc = (
                    "<html><head><title>Foton</title></head><body>"
                    "<h1>Foton</h1>"
                    "<p>Foton jest cząstką elementarną nieposiadającą masy spoczynkowej, będącą nośnikiem oddziaływania elektromagnetycznego.</p>"
                    "<p>Energia fotonu zależy od częstotliwości promieniowania elektromagnetycznego i stałej Plancka.</p>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_doc)))
                self.end_headers()
                self.wfile.write(html_doc)
                return
            self.send_error(404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), WebHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class TestTokenlessUTF8Pipeline(unittest.TestCase):
    def test_full_tokenless_utf8_wernicke_cortex_broca_pipeline(self):
        """
        1) Pełny tor beztokenowy UTF-8: polskie znaki + C++ przechodzą przez
           Wernicke (Byte-Patching + k-WTA 3-5%) -> Modern Hopfield + 16 kolumn Z*
           -> BrocaByteTransformer na CUDA/CPU bez błędów wymiarów i urządzeń.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = BrainConfig(
            language_bootstrap_epochs=0,
            broca_layers=2,
            broca_heads=4,
            broca_dim=256,
            broca_ff_dim=512,
        )
        wernicke = WernickeByteCortex(cfg).to(device)
        cortex = BiologicalAssociativeCortex(cfg, device)
        broca = BrocaByteTransformer(cfg).to(device)

        sample = "Żółć gęślą jaźń — wskaźniki w C++"
        byte_ids, valid_mask = encode_utf8_bytes(
            [sample], max_bytes=cfg.max_input_bytes, patch_size=cfg.patch_stride, device=device
        )
        self.assertEqual(byte_ids.device.type, device.type)
        self.assertEqual(byte_ids.shape[1] % cfg.patch_stride, 0)

        w_out = wernicke.encode_texts([sample], device=device)
        self.assertEqual(tuple(w_out.columns.shape), (1, cfg.num_columns, cfg.column_dim))
        self.assertEqual(tuple(w_out.sdr.shape), (1, cfg.num_neurons))
        self.assertEqual(tuple(w_out.reconstructed_columns.shape), (1, cfg.num_columns, cfg.column_dim))
        self.assertEqual(w_out.active_indices.dtype, torch.int16)
        self.assertEqual(w_out.active_values.dtype, torch.bfloat16)

        # Exact k-WTA sparsity within 3-5%
        active_count = int((w_out.sdr[0] > 0).sum().item())
        self.assertEqual(active_count, cfg.sdr_k)
        self.assertGreaterEqual(w_out.sparsity_pct, 3.0)
        self.assertLessEqual(w_out.sparsity_pct, 5.0)

        # Cortical Modern Hopfield + 16 columns Z*
        fake_bank = w_out.sdr.to(torch.bfloat16)
        thought = cortex.perceive_and_think(
            sensory_columns=w_out.columns,
            sensory_sdr=w_out.sdr,
            engram_bank=fake_bank,
            sdr_to_columns_fn=wernicke.sdr_to_columns,
            steps=4,
        )
        self.assertEqual(tuple(thought.z_star.shape), (1, cfg.num_neurons))
        self.assertEqual(tuple(thought.columns_star.shape), (1, cfg.num_columns, cfg.column_dim))
        self.assertEqual(int((thought.z_star[0] > 0).sum().item()), cfg.sdr_k)
        self.assertEqual(thought.attractor_steps, 4)

        # BrocaByteTransformer conditioned ONLY on Z* streams valid UTF-8
        streamed = list(broca.generate_stream(thought.columns_star, max_new_bytes=36, temperature=0.85))
        joined = "".join(streamed)
        self.assertIsInstance(joined, str)
        self.assertEqual(joined.encode("utf-8").decode("utf-8"), joined)

        # Denoising autoencoder training lowers loss through the SDR bottleneck
        lc = LanguageCortex(cfg=cfg, device=device)
        train_res = lc.train_denoising_sleep(
            texts=[
                "Żółć gęślą jaźń — wskaźniki w C++ przechowują adresy pamięci.",
                "Fotosynteza wytwarza glukozę oraz tlen pod wpływem światła słonecznego.",
                "Pamięć RAM przechowuje dane tymczasowe, a dysk SSD służy do zapisu trwałego.",
                "Stolicą Polski jest Warszawa położona nad rzeką Wisłą.",
            ],
            epochs=6,
        )
        self.assertTrue(train_res["trained"])
        self.assertLess(train_res["loss_end"], train_res["loss_start"])


class TestTensorHippocampusHopfieldSHYAndREM(unittest.TestCase):
    def test_tensor_hippocampus_local_plasticity_shy_and_vectorized_rem(self):
        """
        2) Tensorowy Hipokamp + Modern Hopfield + SHY:
           - bfloat16 engram tensors on GPU/CPU
           - W_rec has strictly zero diagonal
           - prediction_error_end < prediction_error_start
           - pruned_synapses > 0 after SHY
           - REM runs vectorized on GPU and creates new hybrid states
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = BrainConfig(
            language_bootstrap_epochs=0,
            cortex_sleep_epochs=2,
            nrem_sleep_steps=18,
            broca_layers=2,
            broca_heads=4,
            broca_ff_dim=512,
            shy_prune_eps=0.015,
            rem_batch_size=4,
        )
        hippo = Hippocampus(num_neurons=cfg.num_neurons, sdr_k=cfg.sdr_k, device=device)
        cortex = BiologicalAssociativeCortex(cfg, device)
        lc = LanguageCortex(cfg=cfg, device=device)
        engine = SleepEngine(cfg=cfg, cortex=cortex, hippocampus=hippo, device=device, language_cortex=lc)

        stimuli = [
            "Fotosynteza wytwarza tlen oraz glukozę w chloroplastach roślin.",
            "Wskaźnik w C++ przechowuje adres zmiennej w pamięci operacyjnej.",
            "Czarna dziura posiada horyzont zdarzeń, z którego nie ucieka światło.",
            "Chrzest Polski odbył się w 966 roku za panowania Mieszka Pierwszego.",
            "Mitochondria odpowiadają za oddychanie komórkowe i wytwarzanie ATP.",
            "Układ Słoneczny składa się ze Słońca oraz ośmiu planet.",
            "Algorytm to uporządkowany ciąg kroków prowadzących do rozwiązania.",
            "Foton jest bezmasowym kwantem promieniowania elektromagnetycznego.",
        ]
        w_out = lc.perceive_texts(stimuli)
        hippo.record_engrams_batch(
            sdrs=w_out.sdr,
            preview_labels=stimuli,
            category="wiedza",
            day=1,
            sparsity_pct=w_out.sparsity_pct,
        )
        self.assertEqual(hippo.short_term.engrams.dtype, torch.bfloat16)
        self.assertEqual(hippo.short_term.engrams.device.type, device.type)

        # Ensure weak synapses exist in W_rec so SHY deterministically prunes them
        with torch.no_grad():
            cortex.W_rec[0, 1] = 0.005
            cortex.W_rec[1, 2] = 0.006
            cortex.W_rec[2, 3] = 0.007

        report = engine.run_sleep_cycle(night_number=1, wake_texts_for_broca=stimuli)

        # Zero diagonal invariant on W_rec
        diag_max = float(torch.diag(cortex.W_rec).abs().max().item())
        self.assertEqual(diag_max, 0.0)

        # Local Predictive Coding reduced reconstruction error without backprop
        self.assertGreater(report.prediction_error_start, 0.0)
        self.assertLess(report.prediction_error_end, report.prediction_error_start)

        # SHY pruned weak synapses and counted active synapses
        self.assertGreater(report.pruned_synapses, 0)
        self.assertGreaterEqual(report.active_synapses, 0)

        # Short-term cleared, semantic consolidated in bfloat16, REM created vectorized hybrid states
        self.assertEqual(len(hippo.short_term), 0)
        self.assertGreaterEqual(len(hippo.semantic), 4)
        self.assertEqual(hippo.semantic.engrams.dtype, torch.bfloat16)
        self.assertGreater(len(hippo.rem), 0)
        self.assertEqual(len(report.rem_reflections), len(hippo.rem))
        self.assertGreater(len(report.rem_new_links), 0)

        # 2D synaptic projection includes real W_rec edges & active k-WTA flags
        cortex.perceive_and_think(w_out.columns[:1], w_out.sdr[:1], engram_bank=hippo.get_hopfield_bank())
        proj = cortex.compute_2d_projection(max_nodes=32, max_edges=40, engram_bank=hippo.semantic.engrams)
        self.assertGreater(len(proj["nodes"]), 0)
        self.assertTrue(any(n["active"] for n in proj["nodes"]))


class TestAntiCopyAndPersistenceMigration(unittest.TestCase):
    def test_anti_copy_with_erased_preview_labels_and_stochastic_sampling(self):
        """
        3) Kluczowy test anty-kopiujący: po /ucz i /sen wyzeruj wszystkie preview_label
           w Hippocampus i udowodnij, że chat() dalej generuje odpowiedź wyłącznie z Z*
           przez BrocaByteTransformer, a dwa wywołania z temperature > 0 dają różne próbki.
        """
        web_srv = _start_test_web_server()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = BrainConfig(
                    language_bootstrap_epochs=2,
                    cortex_sleep_epochs=2,
                    nrem_sleep_steps=10,
                    broca_layers=2,
                    broca_heads=4,
                    broca_ff_dim=512,
                    broca_max_new_bytes=48,
                )
                brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
                article_url = f"http://127.0.0.1:{web_srv.server_port}/article.html"
                study_res = brain.study_topic_online(article_url, max_rounds=1, consolidate_after=True)
                self.assertEqual(study_res["pages_read"], 1)
                self.assertGreaterEqual(study_res["facts_learned"], 2)
                self.assertGreater(len(brain.hippocampus.semantic), 0)

                # Erase EVERY preview_label across all three banks
                brain.hippocampus.clear_all_preview_labels("")
                for bank in (brain.hippocampus.short_term, brain.hippocampus.semantic, brain.hippocampus.rem):
                    for m in bank.meta:
                        self.assertEqual(m.preview_label, "")

                # Two stochastic calls with temperature > 0 must succeed purely from Z* and differ
                brain.auto_sleep = False
                samples: list[str] = []
                for _ in range(4):
                    ans = brain.chat("Co to jest foton?", temperature=0.90)
                    self.assertIsInstance(ans["reply"], str)
                    self.assertGreater(len(ans["reply"]), 0)
                    self.assertEqual(ans["reply"].encode("utf-8").decode("utf-8"), ans["reply"])
                    samples.append(ans["reply"])

                self.assertGreater(len(set(samples)), 1, f"Expected stochastic non-identical samples, got: {samples}")
        finally:
            web_srv.shutdown()
            web_srv.server_close()

    def test_sparse_sdr_persistence_and_legacy_checkpoint_migration(self):
        """
        4) Persistence, rzadki zapis SDR (int16, bfloat16) oraz automatyczna migracja
           starego pliku tokenowego do *.legacy_<timestamp>.bak bez crasha.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BrainConfig(
                language_bootstrap_epochs=0,
                cortex_sleep_epochs=1,
                nrem_sleep_steps=8,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=512,
            )
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.add_manual_memory("Żółć gęślą jaźń — zapis rzadkiego SDR.", category="wiedza", importance=5)
            brain.add_manual_memory("Wskaźniki w C++ operują na adresach pamięci.", category="wiedza", importance=5)
            brain.sleep_now(visual_pacing=False)

            ckpt_path = brain.save_state()
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["version"], CHECKPOINT_VERSION)
            sem_sparse = saved["hippocampus_engrams"]["semantic"]
            self.assertEqual(sem_sparse["indices"].dtype, torch.int16)
            self.assertEqual(sem_sparse["values"].dtype, torch.bfloat16)
            self.assertEqual(sem_sparse["indices"].shape[1], cfg.sdr_k)

            # Reload into a new SleepingBrain instance
            brain2 = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            self.assertEqual(brain2.day, brain.day)
            self.assertEqual(brain2.total_sleeps, brain.total_sleeps)
            self.assertEqual(len(brain2.hippocampus.semantic), len(brain.hippocampus.semantic))
            self.assertEqual(len(brain2.hippocampus.rem), len(brain.hippocampus.rem))

        # Legacy token-based checkpoint migration test
        with tempfile.TemporaryDirectory() as tmp_legacy:
            state_dir = Path(tmp_legacy) / "brain_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            legacy_pt = state_dir / "brain_weights.pt"
            legacy_json = state_dir / "brain_memory.json"
            torch.save({"concept_cortex": {"id_to_concept": ["stary_token"]}, "language_cortex": {"tokenizer": {}}}, legacy_pt)
            legacy_json.write_text(json.dumps({"embedding_version": "standalone_pytorch_v1", "day": 9}), encoding="utf-8")

            cfg_l = BrainConfig(language_bootstrap_epochs=0, broca_layers=2, broca_heads=4, broca_ff_dim=512)
            migrated = SleepingBrain(cfg=cfg_l, base_dir=tmp_legacy, load_language_mode="sync")
            self.assertEqual(migrated.day, 1)
            self.assertTrue(migrated.language_cortex.ready)
            pt_baks = list(state_dir.glob("brain_weights.pt.legacy_*.bak"))
            json_baks = list(state_dir.glob("brain_memory.json.legacy_*.bak"))
            self.assertEqual(len(pt_baks), 1)
            self.assertEqual(len(json_baks), 1)


class TestWebLearner(unittest.TestCase):
    def test_query_cleaning_relevance_and_direct_url(self):
        self.assertEqual(clean_search_query("sam sie naucz co to sa wskazniki w cpp"), "wskaźniki w C++")
        self.assertEqual(clean_search_query("wzkaźniki w cpp"), "wskaźniki w C++")
        self.assertEqual(clean_search_query("node.ks"), "Node.js")

        doc = WebDocument(
            title="Wskaźnik",
            url="https://example.org",
            text="Wskaźnik w języku C++ przechowuje adres w pamięci programu.",
            source="wikipedia_pl",
        )
        self.assertTrue(is_relevant_doc(doc, "wskazniki w cpp"))
        bad_doc = WebDocument(
            title="Wskaźnik P/E",
            url="https://example.org/pe",
            text="Wskaźnik giełdowy cena do zysku spółki.",
            source="wikipedia_pl",
        )
        self.assertFalse(is_relevant_doc(bad_doc, "wskazniki w cpp"))
        self.assertIsNotNone(extract_lead_definition(doc))

        srv = _start_test_web_server()
        try:
            url = f"http://127.0.0.1:{srv.server_port}/article.html"
            docs = gather_web_sources(url, seen_urls=set(), max_docs=2, timeout=5)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].title, "Foton")
            self.assertIn("cząstką elementarną", docs[0].text)
        finally:
            srv.shutdown()
            srv.server_close()


class TestHTTPServerSSEAndPretrainScript(unittest.TestCase):
    def test_server_endpoints_real_sse_and_pretrain_smoke(self):
        """
        5) Test serwera HTTP/SSE i skryptu pre-trainingu:
           /api/state, realne strumieniowanie /api/chat/stream, /api/sleep,
           /api/memory/add, /api/memory/delete oraz 1-epokowy smoke test scripts/pretrain_broca.py.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BrainConfig(
                language_bootstrap_epochs=1,
                cortex_sleep_epochs=1,
                nrem_sleep_steps=8,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=512,
                broca_max_new_bytes=40,
            )
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            static_dir = Path(__file__).resolve().parent.parent / "static"
            srv_wrap = BrainServer(brain=brain, static_dir=static_dir, port=0)
            httpd = srv_wrap.create_server()
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            base_url = f"http://127.0.0.1:{httpd.server_port}"
            try:
                # GET /api/state
                st = json.loads(urllib.request.urlopen(f"{base_url}/api/state", timeout=10).read().decode("utf-8"))
                lc = st["language_cortex"]
                self.assertTrue(lc["ready"])
                self.assertEqual(lc["num_columns"], cfg.num_columns)
                self.assertGreater(lc["wernicke_params"], 0)
                self.assertGreater(lc["cortex_params"], 0)
                self.assertGreater(lc["broca_params"], 0)
                self.assertGreaterEqual(lc["sdr_sparsity_pct"], 3.0)
                self.assertLessEqual(lc["sdr_sparsity_pct"], 5.0)

                # POST /api/memory/add & /api/memory/delete
                req_add = urllib.request.Request(
                    f"{base_url}/api/memory/add",
                    data=json.dumps({"text": "Żółć gęślą jaźń — test engramu SDR.", "category": "wiedza"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                added = json.loads(urllib.request.urlopen(req_add, timeout=10).read().decode("utf-8"))
                self.assertTrue(added["ok"])
                mem_id = added["memory"]["id"]

                # Add two more so sleep & REM have rich pairs
                for txt in (
                    "Fotosynteza produkuje tlen z dwutlenku węgla i wody.",
                    "Wskaźnik w C++ przechowuje adres obiektu w pamięci.",
                ):
                    urllib.request.urlopen(
                        urllib.request.Request(
                            f"{base_url}/api/memory/add",
                            data=json.dumps({"text": txt, "category": "wiedza"}).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        ),
                        timeout=10,
                    ).read()

                req_del = urllib.request.Request(
                    f"{base_url}/api/memory/delete",
                    data=json.dumps({"id": mem_id}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                deleted = json.loads(urllib.request.urlopen(req_del, timeout=10).read().decode("utf-8"))
                self.assertTrue(deleted["deleted"])

                # POST /api/sleep
                req_sleep = urllib.request.Request(
                    f"{base_url}/api/sleep",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sleep_res = json.loads(urllib.request.urlopen(req_sleep, timeout=30).read().decode("utf-8"))
                self.assertTrue(sleep_res["ok"])
                rep = sleep_res["report"]
                self.assertIn("prediction_error_start", rep)
                self.assertIn("prediction_error_end", rep)
                self.assertIn("pruned_synapses", rep)
                self.assertIn("active_synapses", rep)
                self.assertIn("broca_loss_start", rep)
                self.assertIn("broca_loss_end", rep)

                # POST /api/chat/stream (real UTF-8 streaming from BrocaByteTransformer)
                req_stream = urllib.request.Request(
                    f"{base_url}/api/chat/stream",
                    data=json.dumps({"message": "Żółć gęślą jaźń — co wiesz o fotosyntezie?"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_stream, timeout=20) as resp:
                    raw_sse = resp.read().decode("utf-8")
                events = [
                    json.loads(line[5:].strip())
                    for line in raw_sse.splitlines()
                    if line.startswith("data:")
                ]
                ev_types = [e["type"] for e in events]
                self.assertIn("meta", ev_types)
                self.assertIn("token", ev_types)
                self.assertEqual(ev_types[-1], "done")
                streamed_reply = "".join(e.get("delta", "") for e in events if e["type"] == "token").strip()
                self.assertEqual(streamed_reply, events[-1]["reply"])
            finally:
                httpd.shutdown()
                httpd.server_close()

        # 1-epoch smoke test of scripts/pretrain_broca.py with streaming UTF-8 .txt corpus
        with tempfile.TemporaryDirectory() as tmp_pre:
            corpus_file = Path(tmp_pre) / "pl_corpus.txt"
            corpus_file.write_text(
                "\n".join(
                    [
                        "Żółć gęślą jaźń — polska ortografia i znaki diakrytyczne w UTF-8.",
                        "Wskaźnik w C++ wskazuje adres komórki pamięci operacyjnej RAM.",
                        "Fotosynteza zamienia światło słoneczne, wodę i dwutlenek węgla w glukozę oraz tlen.",
                        "Podczas snu NREM kora asocjacyjna wykonuje lokalne uczenie predykcyjne i homeostazę SHY.",
                        "Faza REM łączy odległe engramy SDR w nowe stany atraktora sieci Hopfielda.",
                    ]
                    * 4
                ),
                encoding="utf-8",
            )
            chunks = list(iter_utf8_text_chunks(corpus_file, chunk_chars=220, max_chunks=10))
            self.assertEqual(len(chunks), 10)

            script_path = Path(__file__).resolve().parent.parent / "scripts" / "pretrain_broca.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--data",
                    str(corpus_file),
                    "--base-dir",
                    str(tmp_pre),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--max-chunks",
                    "16",
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(proc.returncode, 0, f"pretrain_broca failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            self.assertIn("[pretrain_broca] GOTOWE", proc.stdout)
            self.assertTrue((Path(tmp_pre) / "brain_state" / "brain_weights.pt").exists())

    def test_polish_parquet_streaming_and_train_parquet_endpoint(self):
        """
        6) Strumieniowy trening polskiego i gramatyki z Parquet:
           - weryfikacja detekcji i filtrowania realnego train-00000-of-00006.parquet (jeśli istnieje),
           - pełny test POST /api/train_parquet + spadek loss na lokalnym pliku .parquet.
        """
        import time

        import pyarrow as pa
        import pyarrow.parquet as pq

        from brain.language_trainer import inspect_parquet_corpus, iter_parquet_polish_pairs

        real_pq = Path(__file__).resolve().parent.parent / "train-00000-of-00006.parquet"
        if real_pq.exists():
            info = inspect_parquet_corpus(real_pq)
            self.assertTrue(info["exists"])
            self.assertGreater(info["num_rows"], 1000)
            sample_pairs = list(iter_parquet_polish_pairs(real_pq, max_pairs=16))
            self.assertEqual(len(sample_pairs), 16)
            self.assertTrue(any(s == t for s, t in sample_pairs))
            self.assertTrue(any(s != t for s, t in sample_pairs))

        real_pq2 = Path(__file__).resolve().parent.parent / "000_00000.parquet"
        if real_pq2.exists():
            info2 = inspect_parquet_corpus(real_pq2)
            self.assertTrue(info2["exists"])
            self.assertTrue(info2["has_quality_score"])
            self.assertGreater(info2["num_rows"], 1000)
            sample_pairs2 = list(iter_parquet_polish_pairs(real_pq2, max_pairs=12))
            self.assertEqual(len(sample_pairs2), 12)

        with tempfile.TemporaryDirectory() as tmp:
            mini_pq = Path(tmp) / "train-00000-of-00006.parquet"
            mini_pq2 = Path(tmp) / "000_00000.parquet"
            table = pa.table(
                {
                    "id": ["1", "2", "3"],
                    "title": ["Gramatyka", "Składnia", "Ortografia"],
                    "text": [
                        "Język polski należy do rodziny języków słowiańskich i posiada bogatą odmianę przez przypadki. "
                        "Poprawna gramatyka pozwala budować jasne i zrozumiałe zdania w codziennej komunikacji. "
                        "Rzeczowniki odmieniają się przez siedem przypadków oraz dwie liczby.",
                        "Składnia opisuje zasady łączenia wyrazów w spójne wypowiedzenia pojedyncze i złożone. "
                        "Podmiot oraz orzeczenie tworzą główny trzon każdego zdania w języku polskim. "
                        "Właściwa interpunkcja ułatwia szybkie zrozumienie myśli autora tekstu.",
                        "Polska ortografia obejmuje znaki diakrytyczne takie jak ą, ć, ę, ł, ń, ó, ś, ź oraz ż. "
                        "Regularne czytanie dobrej literatury rozwija styl oraz zasób słownictwa.",
                    ],
                }
            )
            pq.write_table(table, mini_pq)

            table2 = pa.table(
                {
                    "text": [
                        "Śmieciowy post na forum z bardzo niską oceną jakości powinien zostać odfiltrowany.",
                        "Polska literatura piękna zachwyca bogactwem metafor, rytmem zdań oraz głębią psychologiczną bohaterów. "
                        "Dobra proza uczy naturalnego szyku wyrazów i precyzyjnego wyrażania myśli.",
                    ],
                    "quality_score": [0.05, 0.96],
                    "language_score": [0.99, 0.999],
                }
            )
            pq.write_table(table2, mini_pq2)

            # Verify quality_score filtering skips row 0 (0.05 < 0.45) and keeps row 1 (0.96)
            filtered_pairs = list(iter_parquet_polish_pairs(mini_pq2, max_pairs=10, min_quality_score=0.45))
            self.assertGreaterEqual(len(filtered_pairs), 2)
            self.assertFalse(any("Śmieciowy post" in s for s, _ in filtered_pairs))

            cfg = BrainConfig(
                language_bootstrap_epochs=0,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=512,
            )
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            static_dir = Path(__file__).resolve().parent.parent / "static"
            srv_wrap = BrainServer(brain=brain, static_dir=static_dir, port=0)
            httpd = srv_wrap.create_server()
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            base_url = f"http://127.0.0.1:{httpd.server_port}"
            try:
                st = json.loads(urllib.request.urlopen(f"{base_url}/api/state", timeout=10).read().decode("utf-8"))
                self.assertTrue(st["parquet_corpus"]["exists"])
                self.assertEqual(st["parquet_corpus"]["file_count"], 2)
                self.assertEqual(st["parquet_corpus"]["total_rows"], 5)

                req_pq = urllib.request.Request(
                    f"{base_url}/api/train_parquet",
                    data=json.dumps({"filename": "__ALL__", "max_steps": 20, "batch_size": 4}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                started = json.loads(urllib.request.urlopen(req_pq, timeout=10).read().decode("utf-8"))
                self.assertTrue(started["ok"])
                self.assertTrue(started["started"])
                self.assertEqual(len(started["files"]), 2)

                # Wait for async background training to complete
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    cur_st = json.loads(urllib.request.urlopen(f"{base_url}/api/state", timeout=10).read().decode("utf-8"))
                    if not cur_st["study"]["active"]:
                        break
                    time.sleep(0.15)
                self.assertFalse(cur_st["study"]["active"])
                self.assertGreater(cur_st["language_cortex"]["trained_steps"], 0)
                self.assertGreater(cur_st["language_cortex"]["fluency"], 0.0)
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_z_star_conditioning_no_cross_attention_collapse_and_dialogue_finetune(self):
        """
        7) Test warunkowania Z* (brak Cross-Attention collapse) + faza dostrajania dialogowego:
           - BASE_HUMAN_DIALOGUES >= 150 par i akceptacja krótkich wejść od 2 znaków ('ok', 'pa', 'siema'),
           - ujednolicona norma RMSNorm na Z* w BiologicalAssociativeCortex.perceive_and_think(),
           - rozkład logitsów Broki dla 'cześć' vs 'co to jest fotosynteza' wyraźnie się różni,
           - dostrajanie dialogowe (train_language_now) obniża loss dla krótkich pytań i Q&A.
        """
        import torch.nn.functional as F

        from brain.language_trainer import BASE_HUMAN_DIALOGUES, build_dialogue_qa_pairs
        from brain.sensory_wernicke import BOS_BYTE

        self.assertGreaterEqual(len(BASE_HUMAN_DIALOGUES), 150)
        qa_pairs = build_dialogue_qa_pairs(include_curriculum=True, min_chars=2)
        self.assertGreaterEqual(len(qa_pairs), 580)
        q_set = {q.lower() for q, _ in qa_pairs}
        self.assertIn("ok", q_set)
        self.assertIn("siema", q_set)
        self.assertIn("cześć", q_set)
        self.assertIn("ale kto", q_set)
        self.assertIn("co to jest teoria względności?", q_set)
        self.assertIn("czym różni się cpu od gpu?", q_set)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = BrainConfig(
                language_bootstrap_epochs=0,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=512,
            )
            self.assertAlmostEqual(cfg.broca_temperature, 0.55, places=2)
            self.assertAlmostEqual(cfg.broca_top_p, 0.85, places=2)
            self.assertAlmostEqual(cfg.broca_repetition_penalty, 1.18, places=2)

            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")

            # Train short dialogue + Q&A phase through Wernicke -> Cortex -> Broca (lr=1e-4, ls=0.05)
            tr = brain.train_language_now(epochs=14)
            self.assertTrue(tr["trained"])
            self.assertGreaterEqual(tr["dialogues"], 350)
            self.assertLess(tr["loss_end"], tr["loss_start"])

            # Verify Z* conditioning on two distinct prompts: "cześć" vs "co to jest fotosynteza"
            w1 = brain.language_cortex.perceive_one("cześć")
            w2 = brain.language_cortex.perceive_one("co to jest fotosynteza")
            t1 = brain.cortex.perceive_and_think(
                w1.columns, w1.sdr, engram_bank=None, sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns
            )
            t2 = brain.cortex.perceive_and_think(
                w2.columns, w2.sdr, engram_bank=None, sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns
            )

            # Unified RMSNorm scale on Z*
            rms1 = float(t1.columns_star.pow(2).mean(dim=-1).sqrt().mean().item())
            rms2 = float(t2.columns_star.pow(2).mean(dim=-1).sqrt().mean().item())
            self.assertGreater(rms1, 0.90)
            self.assertLess(rms1, 1.10)
            self.assertGreater(rms2, 0.90)
            self.assertLess(rms2, 1.10)

            # Z* representations themselves must stay sharply separated (no rank-1 collapse)
            z_cos = float(F.cosine_similarity(t1.columns_star.flatten(1), t2.columns_star.flatten(1)).item())
            self.assertLess(z_cos, 0.72, f"Z* collapsed across distinct prompts! cos={z_cos:.4f}")

            # Broca logits at <BOS> must diverge clearly between "cześć" and "co to jest fotosynteza"
            bos = torch.tensor([[BOS_BYTE]], dtype=torch.long, device=brain.device)
            brain.language_cortex.broca_module.eval()
            with torch.no_grad():
                logits1 = brain.language_cortex.broca_module(bos, t1.columns_star)[:, -1, :].float()
                logits2 = brain.language_cortex.broca_module(bos, t2.columns_star)[:, -1, :].float()
                p1 = F.softmax(logits1 / cfg.broca_temperature, dim=-1)
                p2 = F.softmax(logits2 / cfg.broca_temperature, dim=-1)
                l1_dist = float((p1 - p2).abs().sum().item())
                max_logit_diff = float((logits1 - logits2).abs().max().item())

            self.assertGreater(l1_dist, 0.25, f"Cross-Attention collapse detected! L1(p1, p2)={l1_dist:.4f}")
            self.assertGreater(max_logit_diff, 0.50, f"Broca logits barely depend on Z*! max_diff={max_logit_diff:.4f}")

    def test_working_memory_sdr_self_critique_and_creativity_control(self):
        """
        8) Test rozszerzonej beztokenowej pamięci tymczasowej SDR, wielościeżkowego myślenia
           z pętlą samokrytyki Wernicke → Broca oraz żywej kontroli kreatywności (/api/control).
        """
        from brain.language_cortex import LanguageCortex

        default_cfg = BrainConfig()
        self.assertGreaterEqual(default_cfg.max_short_term_memories, 50000)
        self.assertGreaterEqual(default_cfg.max_semantic_memories, 200000)
        self.assertGreaterEqual(default_cfg.working_chat_window, 128)
        self.assertGreaterEqual(default_cfg.working_chat_buffer, 256)
        self.assertEqual(default_cfg.working_sdr_capacity, 64)
        self.assertGreaterEqual(default_cfg.max_input_bytes, 4096)
        self.assertGreaterEqual(default_cfg.broca_llm_max_new_tokens, 2048)
        self.assertEqual(default_cfg.num_neurons, 12288)
        self.assertFalse(default_cfg.use_local_llm_broca)
        self.assertTrue(default_cfg.self_correction_enabled)

        # Polish quality critic must clearly separate clean Polish from loop garbage
        good_q = LanguageCortex.score_polish_quality(
            "Fotosynteza to proces biologiczny, w którym rośliny wytwarzają glukozę ze światła.",
            creativity_level=0.7,
        )
        loop_q = LanguageCortex.score_polish_quality(
            "ciekawe ciekawe ciekawe i ciekawe ciekawe i ciekawe",
            creativity_level=0.7,
        )
        self.assertGreater(good_q, loop_q + 0.35, f"Expected clean Polish > loop garbage: {good_q} vs {loop_q}")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = BrainConfig(
                language_bootstrap_epochs=1,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=512,
                broca_max_new_bytes=48,
                self_critique_candidates=3,
            )
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            # Direct multi-path deliberation + Wernicke auditory feedback
            delib = brain.language_cortex.deliberate_and_critique(
                user_message="Co to jest foton?",
                cortex=brain.cortex,
                engram_bank=None,
                creativity_level=0.8,
                base_temperature=0.60,
                max_new_bytes=36,
                num_candidates=3,
            )
            self.assertEqual(len(delib["candidates"]), 3)
            labels = [c["label"] for c in delib["candidates"]]
            self.assertEqual(labels, ["Precyzyjna", "Kontekstowa", "Kreatywna"])
            self.assertGreaterEqual(len(delib["critique_notes"]), 4)
            self.assertTrue(any("Słyszę" in n for n in delib["critique_notes"]))

            # Multi-turn chat populates the Prefrontal Working Memory SDR Ring (user + assistant SDRs)
            res1 = brain.chat("Opowiedz o komputerach i procesorach.")
            self.assertGreaterEqual(len(brain._working_sdr_ring), 2)
            self.assertTrue(any("Samokontrola końcowa" in line for line in res1["thinking"]))
            self.assertTrue(any("Pamięć robocza SDR:" in line for line in res1["thinking"]))

            res2 = brain.chat("Dlaczego?")
            self.assertGreaterEqual(len(brain._working_sdr_ring), 4)

            snap = brain.snapshot()
            self.assertEqual(snap["memory"]["working_sdr_count"], len(brain._working_sdr_ring))
            self.assertEqual(snap["limits"]["working_sdr_capacity"], 64)
            self.assertGreaterEqual(snap["limits"]["short_term_memories"], 50000)

            # Conversational online plasticity: giving the brain a name should adapt Broca with ZERO Wernicke drift
            res_name = brain.chat("okej to nadaje ci imie: Krzysztof")
            self.assertTrue(any("plastyczność konwersacyjna" in line.lower() for line in res_name["thinking"]))
            self.assertGreater(brain.language_cortex._known_q_sdrs.shape[0], 0)
            self.assertFalse(brain._is_question_or_command("okej to nadaje ci imie: Krzysztof"))
            self.assertTrue(brain._is_question_or_command("jak masz na imie"))

            # Live creativity update via HTTP /api/control
            static_dir = Path(__file__).resolve().parent.parent / "static"
            srv_wrap = BrainServer(brain=brain, static_dir=static_dir, port=0)
            httpd = srv_wrap.create_server()
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            base_url = f"http://127.0.0.1:{httpd.server_port}"
            try:
                req = urllib.request.Request(
                    f"{base_url}/api/control",
                    data=json.dumps({"creativity_level": 0.92}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                ctrl = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
                self.assertTrue(ctrl["ok"])
                self.assertAlmostEqual(brain.creativity_level, 0.92, places=2)
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_hybrid_local_broca_llm_prompt_and_config(self):
        """
        9) Test integracji hybrydowego obszaru Broki (LocalPolishBrocaLLM):
           - dynamiczne budowanie kontekstu biologicznego (Dzień, kreatywność, pamięć SDR, REM)
           - wieloturowa historia rozmowy bez stałych tekstów w kodzie.
        """
        from brain.broca_llm import LocalPolishBrocaLLM

        with tempfile.TemporaryDirectory() as tmp:
            cfg = BrainConfig(use_local_llm_broca=True)
            llm = LocalPolishBrocaLLM(cfg=cfg, device=torch.device("cpu"), lora_dir=Path(tmp) / "lora")
            self.assertEqual(llm.model_id, "MassivDash/Bielik-4.5B-v3-heretic")
            self.assertGreaterEqual(cfg.broca_llm_max_new_tokens, 2048)
            sys_p = llm.build_system_prompt(
                day=5,
                total_sleeps=4,
                creativity_level=0.85,
                recalled_semantic=["Fotosynteza zachodzi w chloroplastach roślin."],
                recalled_hippo=["Użytkownik nadał mi imię Krzysztof."],
                recalled_rem=["Skojarzenie REM między światłem a pamięcią."],
                sdr_sparsity_pct=4.0,
            )
            self.assertIn("Dzień 5", sys_p)
            self.assertIn("85%", sys_p)
            self.assertIn("Krzysztof", sys_p)
            self.assertIn("Fotosynteza", sys_p)

            msgs = llm._build_chat_messages(
                sys_p,
                [{"role": "user", "content": "Siema co tam!"}, {"role": "assistant", "content": "Siema! Wszystko gra."}],
                "SUEPR to co robimy",
            )
            self.assertEqual(len(msgs), 4)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[-1]["content"], "SUEPR to co robimy")

    def test_living_brain_neurogenesis_curiosity_neuromod_and_budget_compaction(self):
        """
        10) Test Living Brain (M1/M2/M4 bez backprop):
            - zero requires_grad (wyłącznie lokalna reguła Oja-Hebba + Predictive Coding)
            - dynamiczna neurogeneza (wzrost L1/L2/L3/Assoc/M4 bez sztywnego limitu)
            - zaskoczenie (L3->L2->L1), predictive suppression, ciekawość M4 i eksploracja nudy
            - neuromodulacja NA/DA/ACh, offline sleep replay (M2->M1) i zmęczenie neuronów
            - dynamiczny budżet pamięci (68 MB - 32 GB), przełączanie GPU/RAM oraz natychmiastowe
              przycinanie i kompaktowanie tensorów przy obniżeniu suwaka!
        """
        from brain.living_brain import LivingBrainEngine

        lb = LivingBrainEngine(
            sensory_dim=256,
            target_device="gpu" if torch.cuda.is_available() else "ram",
            memory_budget_mb=2048.0,
            start_background_clock=False,
        )
        # 1) Strictly zero backpropagation parameters anywhere
        for attr in ("W_in_L1", "G_10", "W_12", "G_21", "W_23", "G_32", "W_3a", "W_assoc_rec", "m4_prototypes"):
            self.assertFalse(getattr(lb, attr).requires_grad)

        init_neurons = lb.total_neurons_count()
        init_mb = lb.living_memory_mb()

        # 2) Drive novel surprising stimuli -> triggers M2 encoding + structural neurogenesis
        tel_list = []
        for i in range(6):
            stim = torch.randn(256)
            tel_list.append(lb.cognitive_step(sensory_vec=stim, external_valence=0.4, allow_growth=True))
        self.assertGreater(lb.total_neurons_count(), init_neurons, "Neurogenesis should grow neuron count dynamically")
        self.assertGreater(int(lb.m2_assoc_traces.shape[0]), 0, "Surprising episodes should encode into M2")
        self.assertTrue(any(t.surprise > 0.0 for t in tel_list))
        self.assertGreater(lb.last_telemetry.mean_fatigue, 0.0)

        # 3) Explicit growth pushes memory up, then lowering slider to 68 MB compacts tensors immediately!
        for _ in range(4):
            lb.grow_structure(add_l1=512, add_l2=384, add_l3=256, add_assoc=384, add_m4=64)
        grown_neurons = lb.total_neurons_count()
        grown_mb = lb.living_memory_mb()
        self.assertGreater(grown_neurons, init_neurons)
        self.assertGreater(grown_mb, init_mb)

        # Test sleep replay M2 -> M1 + SHY pruning + fatigue reset
        rep = lb.sleep_consolidate_replay(replay_steps=6)
        self.assertGreater(rep["replayed_episodes"], 0)
        self.assertEqual(float(lb.fatigue_l1.abs().max().item()), 0.0)

        # Test RAM/GPU device switch + lowering budget to 68 MB triggers controlled compaction
        res_ram = lb.set_memory_budget(budget_mb=68.0, target_device="ram")
        self.assertEqual(res_ram["target_device"], "ram")
        self.assertLessEqual(res_ram["living_mb"], 68.0)
        if torch.cuda.is_available():
            res_gpu = lb.set_memory_budget(budget_mb=1024.0, target_device="gpu")
            self.assertEqual(res_gpu["target_device"], "gpu")

    def test_priority1_living_projection_neuromod_and_pinned_core(self):
        """
        11) Test Priorytetu 1:
            - bezpośrednia adaptacyjna projekcja rosnącego LivingBrainEngine (state_assoc + top-3 M2) do Z*
              (z zachowaniem RMSNorm i bez awarii przy grow_structure()),
            - sterowanie krokami atraktora i temperaturą przez NA oraz ścieżką Kreatywną przez ciekawość M4,
            - dopaminowe lekkie wzmocnienie online przy czystej samokrytyce + aktywny zegar tła,
            - pełna ochrona pinned=True w Hippocampus (sen + capacity), LivingBrainEngine (m2_pinned + compaction)
              oraz LanguageCortex (_learned_qa_replay przy >200 wpisach).
        """
        import time

        from brain.hippocampus import EngramMeta, Hippocampus
        from brain.living_brain import LivingBrainEngine, LivingStepTelemetry, NeuromodulatorState

        # 1) LivingBrainEngine -> Z* adaptive projection across neurogenesis growth
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BrainConfig(
                language_bootstrap_epochs=0,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=512,
                broca_max_new_bytes=36,
            )
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False
            try:
                # Background spontaneous clock must be running automatically
                self.assertIsNotNone(brain.living_brain._bg_thread)
                self.assertTrue(brain.living_brain._bg_thread.is_alive())
                t0_ticks = brain.living_brain.tick_count
                time.sleep(0.8)
                self.assertGreater(brain.living_brain.tick_count, t0_ticks)

                w = brain.language_cortex.perceive_one("Co to jest fotosynteza?")
                base_thought = brain.cortex.perceive_and_think(
                    w.columns,
                    w.sdr,
                    engram_bank=None,
                    sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                    steps=3,
                )

                # Drive LivingBrain with stimuli + M2 traces, then project into Z*
                for _ in range(4):
                    brain.living_brain.cognitive_step(
                        sensory_vec=torch.randn(512),
                        external_valence=0.4,
                        allow_growth=False,
                    )
                assoc_v1 = brain.living_brain.get_cortical_association_vector(top_k_m2=3)
                self.assertEqual(int(assoc_v1.numel()), brain.living_brain.n_assoc)

                th_lb1 = brain.cortex.perceive_and_think(
                    w.columns,
                    w.sdr,
                    engram_bank=None,
                    sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                    steps=3,
                    living_assoc_vec=assoc_v1,
                    living_blend=0.10,
                )
                diff1 = float((th_lb1.columns_star - base_thought.columns_star).abs().mean().item())
                self.assertGreater(diff1, 1e-3, "LivingBrain projection should genuinely modify Z*")
                rms1 = float(th_lb1.columns_star.pow(2).mean(dim=-1).sqrt().mean().item())
                self.assertGreater(rms1, 0.90)
                self.assertLess(rms1, 1.10)

                # Grow LivingBrain structure dynamically -> n_assoc changes, projection stays size-independent!
                old_assoc_n = brain.living_brain.n_assoc
                born = brain.living_brain.grow_structure(add_l1=64, add_l2=48, add_l3=32, add_assoc=64, add_m4=16)
                self.assertGreater(born, 0)
                self.assertEqual(brain.living_brain.n_assoc, old_assoc_n + 64)
                brain.living_brain.cognitive_step(sensory_vec=torch.randn(512), external_valence=0.5, allow_growth=False)
                assoc_v2 = brain.living_brain.get_cortical_association_vector(top_k_m2=3)
                self.assertEqual(int(assoc_v2.numel()), old_assoc_n + 64)
                th_lb2 = brain.cortex.perceive_and_think(
                    w.columns,
                    w.sdr,
                    engram_bank=None,
                    sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                    steps=3,
                    living_assoc_vec=assoc_v2,
                    living_blend=0.10,
                )
                self.assertEqual(tuple(th_lb2.columns_star.shape), tuple(base_thought.columns_star.shape))

                # 2) Neuromodulator control: High NA -> 6 attractor steps + lower T; High Curiosity -> higher Creative T
                high_na = NeuromodulatorState(noradrenaline=2.2, dopamine=0.5, acetylcholine=0.8)
                high_pe_tel = LivingStepTelemetry(surprise=0.80, intrinsic_curiosity=0.10, boredom_exploration=False)
                th_na = brain.cortex.perceive_and_think(
                    w.columns,
                    w.sdr,
                    engram_bank=None,
                    sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                    steps=3,
                    neuromod=high_na,
                    living_telemetry=high_pe_tel,
                )
                self.assertEqual(th_na.attractor_steps, 6)

                delib_na = brain.language_cortex.deliberate_and_critique(
                    user_message="Co to jest fotosynteza?",
                    cortex=brain.cortex,
                    engram_bank=None,
                    creativity_level=0.5,
                    base_temperature=0.62,
                    max_new_bytes=28,
                    num_candidates=3,
                    living_assoc_vec=assoc_v2,
                    neuromod=high_na,
                    living_telemetry=high_pe_tel,
                )
                self.assertEqual(delib_na["thought"].attractor_steps, 6)
                prec_na_t = next(c["temperature"] for c in delib_na["candidates"] if c["label"] == "Precyzyjna")

                low_na = NeuromodulatorState(noradrenaline=0.30, dopamine=0.7, acetylcholine=0.4)
                high_cur_tel = LivingStepTelemetry(surprise=0.15, intrinsic_curiosity=0.95, boredom_exploration=True)
                delib_cur = brain.language_cortex.deliberate_and_critique(
                    user_message="Co to jest fotosynteza?",
                    cortex=brain.cortex,
                    engram_bank=None,
                    creativity_level=0.5,
                    base_temperature=0.62,
                    max_new_bytes=28,
                    num_candidates=3,
                    living_assoc_vec=assoc_v2,
                    neuromod=low_na,
                    living_telemetry=high_cur_tel,
                )
                prec_low_t = next(c["temperature"] for c in delib_cur["candidates"] if c["label"] == "Precyzyjna")
                cre_na_t = next(c["temperature"] for c in delib_na["candidates"] if c["label"] == "Kreatywna")
                cre_cur_t = next(c["temperature"] for c in delib_cur["candidates"] if c["label"] == "Kreatywna")
                self.assertLess(prec_na_t, prec_low_t, "High NA should lower sampling temperature for factual precision")
                self.assertGreater(cre_cur_t, cre_na_t, "High curiosity / boredom exploration should raise Creative temperature")

                # 3) Pinned identity memories via chat ("nadaję ci imię: Krzysztof") survive >200 replay writes & compaction
                brain.chat("nadaję ci imię: Krzysztof")
                self.assertIn("jak masz na imię", brain.language_cortex._pinned_qa_keys)
                self.assertTrue(any(m.pinned for m in brain.hippocampus.short_term.meta))
                self.assertGreater(int(brain.living_brain.m2_pinned.sum().item()), 0)

                # Push >210 unpinned pairs into _learned_qa_replay — pinned identity pairs MUST NOT be evicted!
                flood_pairs = [(f"pytanie testowe numer {i}", f"To jest odpowiedź testowa numer {i}.") for i in range(215)]
                brain.language_cortex.adapt_qa_pairs_online(flood_pairs[:2], cortex=brain.cortex, steps=4, pinned=False, lightweight=True)
                # Directly exercise buffer cap logic with 215 unpinned entries appended via adapt_qa_pairs_online
                with brain.language_cortex._op_lock:
                    brain.language_cortex._learned_qa_replay.extend(flood_pairs)
                brain.language_cortex.adapt_qa_pairs_online(
                    [("kolejne pytanie testowe", "To jest kolejna odpowiedź testowa.")],
                    cortex=brain.cortex,
                    steps=4,
                    pinned=False,
                    lightweight=True,
                )
                replay_qs = {q.lower() for q, _ in brain.language_cortex._learned_qa_replay}
                self.assertIn("jak masz na imię", replay_qs)
                self.assertIn("jak się nazywasz", replay_qs)

                # Checkpoint save/load preserves pinned flags across Hippocampus, LivingBrain & LanguageCortex
                brain.save_state()
                brain2 = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
                try:
                    self.assertIn("jak masz na imię", brain2.language_cortex._pinned_qa_keys)
                    self.assertTrue(any(m.pinned for m in brain2.hippocampus.short_term.meta))
                    self.assertGreater(int(brain2.living_brain.m2_pinned.sum().item()), 0)
                finally:
                    brain2.living_brain.stop_spontaneous_clock()
            finally:
                brain.living_brain.stop_spontaneous_clock()

        # 4) Hippocampus pinned immunity: _enforce_capacity() & consolidate_nrem_light_gpu()
        hippo = Hippocampus(num_neurons=128, sdr_k=8, max_short_term_memories=20, max_semantic_memories=4, device=torch.device("cpu"))
        pinned_sdr = torch.zeros(128)
        pinned_sdr[:8] = 1.0
        hippo.semantic.add_batch(
            sdrs=pinned_sdr.unsqueeze(0),
            metas=[EngramMeta(id="pin01", category="zasada", preview_label="Moje imię to Krzysztof", pinned=True)],
            strengths=[0.05],  # below prune_threshold=0.18!
            importances=[1.0],  # low importance!
            day=1,
        )
        # Add 6 high-priority unpinned engrams to exceed max_semantic_memories=4
        for i in range(6):
            v = torch.zeros(128)
            v[10 + i * 8 : 18 + i * 8] = 1.0
            hippo.semantic.add_batch(
                sdrs=v.unsqueeze(0),
                metas=[EngramMeta(id=f"unp{i}", category="wiedza", preview_label=f"Fakt {i}", pinned=False)],
                strengths=[5.0],
                importances=[5.0],
                day=2,
            )
        self.assertIn("pin01", [m.id for m in hippo.semantic.meta], "Pinned engram must survive _enforce_capacity()")
        hippo.consolidate_nrem_light_gpu(decay=0.5, prune_threshold=0.5, current_day=10)
        pin_idx = [m.id for m in hippo.semantic.meta].index("pin01")
        self.assertAlmostEqual(float(hippo.semantic.strengths[pin_idx].item()), 0.05, places=4)

        # 5) LivingBrainEngine m2_pinned immunity during budget compaction to 68 MB & _prune_weak_episodes()
        lb = LivingBrainEngine(sensory_dim=256, target_device="ram", memory_budget_mb=2048.0, start_background_clock=False)
        lb.cognitive_step(sensory_vec=torch.randn(256), external_valence=0.5, allow_growth=False, pinned=True)
        lb.cognitive_step(sensory_vec=torch.randn(256), external_valence=0.5, allow_growth=False, pinned=True)
        pinned_before = int(lb.m2_pinned.sum().item())
        self.assertGreaterEqual(pinned_before, 1)
        for _ in range(18):
            lb.cognitive_step(sensory_vec=torch.randn(256), external_valence=0.1, allow_growth=False, pinned=False)
        for _ in range(4):
            lb.grow_structure(add_l1=512, add_l2=384, add_l3=256, add_assoc=384, add_m4=64)
        lb.set_memory_budget(budget_mb=68.0, target_device="ram")
        lb._prune_weak_episodes(fraction=0.80)
        self.assertEqual(int(lb.m2_pinned.sum().item()), pinned_before, "m2_pinned episodes must survive compaction & pruning!")


if __name__ == "__main__":
    unittest.main()
