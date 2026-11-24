from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from brain.agent import SleepingBrain
from brain.config import BrainConfig
from brain.hippocampus import Hippocampus, _cosine
from brain.language_cortex import LanguageCortex, _fallback_hash_embed
from brain.neural_cortex import AssociativeConceptCortex
from brain.neural_language import CorticalSemanticEncoder, CorticalTransformerLM, SubwordTokenizer
from brain.sleep_engine import SleepEngine
from brain.web_learner import WebDocument, clean_search_query, extract_lead_definition, gather_web_sources, is_relevant_doc
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
                    "<p>Foton jest cząstką elementarną nieposiadającą masy spoczynkowej, będącą nośnikiem oddziaływaniem elektromagnetycznym.</p>"
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


class TestNeuralLanguageModels(unittest.TestCase):
    def test_tokenizer_lossless_polish_and_code(self):
        tok = SubwordTokenizer(max_vocab_size=2048)
        sample = "Żółć gęślą jaźń — int* ptr = &x; // C++ wskaźnik"
        tok.learn_texts([sample])
        ids = tok.encode(sample, max_len=128)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, sample)
        self.assertGreater(tok.vocab_used, SubwordTokenizer.VOCAB_OFFSET)

        state = tok.export_state()
        tok2 = SubwordTokenizer(max_vocab_size=2048)
        tok2.import_state(state)
        self.assertEqual(tok2.encode(sample, max_len=128), ids)

    def test_semantic_encoder_and_transformer_gpu_training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = SubwordTokenizer(max_vocab_size=4096)
        pairs = [
            ("kunktator", "Kunktator to osoba powolna i ostrożna w działaniu."),
            ("wskaźnik C++", "Wskaźnik w C++ przechowuje adres zmiennej w pamięci."),
            ("foton", "Foton jest cząstką elementarną bez masy spoczynkowej."),
            ("czarna dziura", "Czarna dziura ma horyzont zdarzeń, z którego nie ucieka światło."),
        ]
        tok.learn_texts([p[0] + " " + p[1] for p in pairs])

        enc = CorticalSemanticEncoder(vocab_size=4096, ngram_buckets=1024, d_model=128).to(device)
        vecs = enc.encode_texts(["kunktatora", "Kunktator to osoba powolna.", "kompilator Rust"], tok)
        self.assertGreater(_cosine(vecs[0], vecs[1]), _cosine(vecs[0], vecs[2]))

        enc_s, enc_e = enc.train_contrastive(pairs, tok, epochs=10, lr=2e-3)
        self.assertLess(enc_e, enc_s)

        lm = CorticalTransformerLM(vocab_size=4096, d_model=128, n_layers=2, n_heads=4, ff_dim=256, max_seq_len=96).to(device)
        seqs = [f"<Q> {q} <A> {a}" for q, a in pairs]
        lm_s, lm_e = lm.train_sleep_replay(seqs, tok, epochs=12, lr=2e-3)
        self.assertLess(lm_e, lm_s)

        exported = lm.export_state()
        lm2 = CorticalTransformerLM(vocab_size=4096, d_model=128, n_layers=2, n_heads=4, ff_dim=256, max_seq_len=96).to(device)
        lm2.import_state(exported)
        self.assertEqual(lm2.total_parameters(), lm.total_parameters())


class TestLanguageCortexStandalone(unittest.TestCase):
    def test_extraction_update_no_hallucination_and_streaming(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = BrainConfig(cortex_sleep_epochs=8)
        lc = LanguageCortex(cfg=cfg, device=device)
        self.assertTrue(lc.load_sync())
        self.assertGreater(lc.total_parameters(), 1_000_000)

        # Deterministic fallback hash check
        self.assertEqual(_fallback_hash_embed("wskaźniki C++"), _fallback_hash_embed("wskaźniki C++"))

        # Questions without '?' or diacritics must NOT be stored
        self.assertEqual(lc.extract_memories_from_turn("jak dziala pamiec")["memories"], [])
        self.assertEqual(lc.extract_memories_from_turn("czym sa wskazniki w cpp")["memories"], [])
        self.assertEqual(lc.extract_memories_from_turn("ale to nie prawda")["memories"], [])

        # Declarative user info is extracted cleanly
        ext = lc.extract_memories_from_turn("Mam na imię Janek, mieszkam w Krakowie i interesuję się fizyką.")
        texts = [m["text"] for m in ext["memories"]]
        self.assertTrue(any("Janek" in t for t in texts))
        self.assertTrue(any("Krakowie" in t for t in texts))

        # Contradiction / update detection
        idx = lc.check_memory_update("Użytkownik mieszka w Gdańsku.", ["Użytkownik ma na imię Janek.", "Użytkownik mieszka w Krakowie."])
        self.assertEqual(idx, 1)

        # Anti-hallucination gate on unknown subject
        unknown_reply = lc.chat_reply("Kto to był Tamerlan?", [], [], [], [], [], 1, 0, 100.0, "AWAKE")
        self.assertIn("/ucz", unknown_reply)

        # Conversational capability with typos & short follow-ups must NOT suggest '/ucz ...'
        cap_reply = lc.chat_reply("o czy moge z toba rozmiac", [], [], [], [], [], 1, 0, 100.0, "AWAKE")
        self.assertNotIn("/ucz o czy", cap_reply.lower())
        self.assertIn("/szkola", cap_reply.lower())

        follow_reply = lc.chat_reply("ale co", [{"role": "assistant", "content": "Cześć!"}], [], [], [], [], 1, 0, 100.0, "AWAKE")
        self.assertNotIn("/ucz ale co", follow_reply.lower())

        # Elementary school math solver
        self.assertIn("56", lc.chat_reply("ile to jest 7 * 8", [], [], [], [], [], 1, 0, 100.0, "AWAKE"))
        self.assertIn("50", lc.chat_reply("ile to 25% z 200", [], [], [], [], [], 1, 0, 100.0, "AWAKE"))
        self.assertIn("12", lc.chat_reply("pierwiastek z 144", [], [], [], [], [], 1, 0, 100.0, "AWAKE"))

        # Streaming reply grounded in memory
        chunks = list(
            lc.chat_reply_stream(
                "no to kto to jest Kunktator",
                [],
                [],
                ["Kunktator to osoba powolna i ostrożna w działaniu; przydomek rzymskiego wodza Fabiusza Maksimusa."],
                [],
                [],
                1,
                0,
                100.0,
                "AWAKE",
            )
        )
        full = "".join(chunks)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("powolna", full.lower())


class TestHippocampusAndCortex(unittest.TestCase):
    def test_polish_inflection_search_and_crud(self):
        hippo = Hippocampus()
        m1 = hippo.record_memory("Kunktator to osoba powolna i ostrożna w działaniu.", "nauka_z_sieci", 5, day=1, topic="Kunktator")
        hippo.record_memory("Wskaźnik w C++ przechowuje adres zmiennej w pamięci.", "nauka_z_sieci", 5, day=1, topic="wskaźniki C++")

        res1 = hippo.search_memories([], "opowiedz o kunktatora", current_day=1)
        self.assertTrue(any("Kunktator" in m.text for m in res1["hippocampus"]))

        res2 = hippo.search_memories([], "wskazniki w cpp", current_day=1)
        self.assertTrue(any("Wskaźnik" in m.text for m in res2["hippocampus"]))

        hippo.studied_topics["kunktator"] = {"topic": "Kunktator", "mastery": 90}
        self.assertTrue(hippo.delete_memory(m1.id))
        self.assertFalse(hippo.delete_memory("nonexistent"))
        self.assertTrue(hippo.delete_topic("Kunktator"))
        self.assertNotIn("kunktator", hippo.studied_topics)

    def test_concept_cortex_gpu_ewc_rem_and_projection(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = BrainConfig(nrem_concept_epochs=45, rem_similarity_threshold=-1.0)
        cortex = AssociativeConceptCortex(cfg, device)
        hippo = Hippocampus()
        hippo.record_fact("kunktator", "określa", "fabiusz", day=1)
        hippo.record_fact("fabiusz", "był", "konsulem", day=1)
        hippo.record_fact("wskaźnik", "przechowuje", "adres", day=1)

        engine = SleepEngine(cfg=cfg, concept_cortex=cortex, hippocampus=hippo, device=device)
        rep = engine.run_sleep_cycle(night_number=1)
        self.assertLess(rep.concept_loss_end, rep.concept_loss_start)
        self.assertGreaterEqual(len(rep.rem_new_links), 1)

        self.assertEqual(cortex.resolve_concept("kunktatora"), "kunktator")
        self.assertEqual(cortex.resolve_concept("wskazniki"), "wskaźnik")
        proj = cortex.compute_2d_projection(limit=10, concepts=["kunktator", "fabiusz", "konsulem"])
        self.assertEqual(set(proj.keys()), {"kunktator", "fabiusz", "konsulem"})


class TestWebLearner(unittest.TestCase):
    def test_query_cleaning_relevance_and_direct_url(self):
        self.assertEqual(clean_search_query("sam sie naucz co to sa wskazniki w cpp"), "wskaźniki w C++")
        self.assertEqual(clean_search_query("wzkaźniki w cpp"), "wskaźniki w C++")
        self.assertEqual(clean_search_query("node.ks"), "Node.js")

        doc = WebDocument(title="Wskaźnik", url="https://example.org", text="Wskaźnik w języku C++ przechowuje adres w pamięci programu.", source="wikipedia_pl")
        self.assertTrue(is_relevant_doc(doc, "wskazniki w cpp"))
        bad_doc = WebDocument(title="Wskaźnik P/E", url="https://example.org/pe", text="Wskaźnik giełdowy cena do zysku spółki.", source="wikipedia_pl")
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


class TestSleepingBrainAndServer(unittest.TestCase):
    def test_real_state_migration_and_server_sse(self):
        # 1) Read-only copy of user's real brain_state migrates & answers in Polish without Ollama
        real_state = Path(__file__).resolve().parent.parent / "brain_state"
        if (real_state / "brain_memory.json").exists() and (real_state / "brain_weights.pt").exists():
            with tempfile.TemporaryDirectory() as tmp_copy:
                dst_state = Path(tmp_copy) / "brain_state"
                dst_state.mkdir(parents=True, exist_ok=True)
                shutil.copy2(real_state / "brain_memory.json", dst_state / "brain_memory.json")
                shutil.copy2(real_state / "brain_weights.pt", dst_state / "brain_weights.pt")
                b_real = SleepingBrain(base_dir=tmp_copy, load_language_mode="sync")
                self.assertGreaterEqual(b_real.day, 1)
                snap_real = b_real.snapshot()
                self.assertIn("nodes", snap_real["graph"])
                # Anti-hallucination on unlearned topic when Auto-web is disabled
                b_real.auto_web = False
                unk_real = b_real.chat("co to jest kwark powabny")
                self.assertIn("/ucz", unk_real["reply"])
                # Conversational Polish small-talk & typo tolerance
                wow_res = b_real.chat("wow")
                self.assertNotIn("/ucz", wow_res["reply"])
                compliment_res = b_real.chat("moadry jestes")
                self.assertNotIn("/szkola", compliment_res["reply"])

        # 2) Direct URL study + sleep GPU loss + BrainServer SSE & CRUD
        web_srv = _start_test_web_server()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = BrainConfig(nrem_concept_epochs=25, cortex_sleep_epochs=8)
                brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
                brain.working_chat = [
                    {"role": "user", "content": "Cześć, mam na imię Janek."},
                    {"role": "assistant", "content": "Cześć Janek!"},
                    {"role": "user", "content": "Co to foton?"},
                    {"role": "assistant", "content": "Nie wiem dokładnie co to foton — użyj /ucz foton."},
                ]
                article_url = f"http://127.0.0.1:{web_srv.server_port}/article.html"
                res = brain.study_topic_online(article_url, max_rounds=1, consolidate_after=True)
                self.assertEqual(res["pages_read"], 1)
                self.assertGreaterEqual(res["facts_learned"], 2)
                self.assertIsNotNone(res["sleep_report"])
                self.assertTrue(res["sleep_report"]["lora_trained"])
                self.assertLess(res["sleep_report"]["lora_loss_end"], res["sleep_report"]["lora_loss_start"])

                contents = [t["content"] for t in brain.working_chat]
                self.assertTrue(any("Janek" in c for c in contents))
                self.assertFalse(any("Nie wiem dokładnie co to foton" in c for c in contents))

                static_dir = Path(__file__).resolve().parent.parent / "static"
                server_wrapper = BrainServer(brain=brain, static_dir=static_dir, port=0)
                httpd = server_wrapper.create_server()
                t = threading.Thread(target=httpd.serve_forever, daemon=True)
                t.start()
                srv_url = f"http://127.0.0.1:{httpd.server_port}"
                try:
                    st = json.loads(urllib.request.urlopen(f"{srv_url}/api/state", timeout=5).read().decode("utf-8"))
                    self.assertTrue(st["language_cortex"]["ready"])
                    self.assertGreater(st["language_cortex"]["total_params"], 1_000_000)

                    exp = json.loads(urllib.request.urlopen(f"{srv_url}/api/export", timeout=5).read().decode("utf-8"))
                    self.assertIn("semantic_memories", exp)

                    req_add = urllib.request.Request(
                        f"{srv_url}/api/memory/add",
                        data=json.dumps({"text": "Zasada: zawsze używaj TypeScript.", "category": "zasada"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    added = json.loads(urllib.request.urlopen(req_add, timeout=5).read().decode("utf-8"))
                    self.assertTrue(added["ok"])
                    mem_id = added["memory"]["id"]

                    req_del = urllib.request.Request(
                        f"{srv_url}/api/memory/delete",
                        data=json.dumps({"id": mem_id}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    deleted = json.loads(urllib.request.urlopen(req_del, timeout=5).read().decode("utf-8"))
                    self.assertTrue(deleted["deleted"])

                    req_stream = urllib.request.Request(
                        f"{srv_url}/api/chat/stream",
                        data=json.dumps({"message": "co to jest foton"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_stream, timeout=10) as resp:
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
                    self.assertIn("cząstką", events[-1]["reply"].lower())

                    # Screenshot regression: 'ale co' must have 0 unrelated semantic memory pills
                    ale_co = brain.chat("ale co")
                    self.assertEqual(ale_co["recalled_memories"]["semantic"], [])
                    self.assertNotIn("/ucz ale co", ale_co["reply"].lower())

                    cap_chat = brain.chat("o czy moge z toba rozmiac")
                    self.assertNotIn("/ucz o czy", cap_chat["reply"].lower())

                    # POST /api/curriculum (Elementary School pack + GPU training + idempotency)
                    req_curr = urllib.request.Request(
                        f"{srv_url}/api/curriculum",
                        data=json.dumps({"deep_web": False}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    curr = json.loads(urllib.request.urlopen(req_curr, timeout=30).read().decode("utf-8"))
                    self.assertTrue(curr["ok"])
                    self.assertGreaterEqual(curr["added_facts"], 80)
                    self.assertGreaterEqual(curr["added_triples"], 70)
                    self.assertTrue(curr["sleep_report"]["lora_trained"])

                    # Re-applying /api/curriculum is idempotent (0 duplicates)
                    curr2 = json.loads(urllib.request.urlopen(
                        urllib.request.Request(
                            f"{srv_url}/api/curriculum",
                            data=json.dumps({"deep_web": False}).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        ),
                        timeout=15,
                    ).read().decode("utf-8"))
                    self.assertEqual(curr2["added_facts"], 0)

                    # Elementary school Q&A after GPU consolidation
                    self.assertIn("warszawa", brain.chat("jaka jest stolica Polski")["reply"].lower())
                    foto = brain.chat("na czym polega fotosynteza")["reply"].lower()
                    self.assertTrue("tlen" in foto or "glukoz" in foto)
                    self.assertTrue(any(w in brain.chat("ile planet ma Układ Słoneczny")["reply"].lower() for w in ("ośmiu", "8", "merkury")))
                    self.assertIn("966", brain.chat("kiedy był chrzest Polski")["reply"].lower())
                finally:
                    httpd.shutdown()
                    httpd.server_close()
        finally:
            web_srv.shutdown()
            web_srv.server_close()


if __name__ == "__main__":
    unittest.main()
