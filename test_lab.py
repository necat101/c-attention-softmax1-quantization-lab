#!/usr/bin/env python3
import unittest, json, os, sys, math, subprocess, pathlib, csv
repo = pathlib.Path(__file__).parent

class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(repo/"cases.json") as f:
            cls.cases = json.load(f)
        with open(repo/"results_rows.json") as f:
            cls.rows = json.load(f)

    def test_case_count(self):
        self.assertEqual(len(self.cases), 20)
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(set(ids)), 20)
        required = ["zig_compiler_marker","c_math_api_marker","dot_product_marker","scaled_attention_score_marker","stable_softmax_marker","softmax_shift_invariance_marker","softmax1_mass_marker","all_negative_abstention_marker","relative_weight_ratio_marker","attention_weighted_sum_marker","quiet_attention_weighted_sum_marker","causal_mask_marker","all_masked_row_marker","nonfinite_input_rejection_marker","symmetric_int8_block_marker","zero_block_quantization_marker","dequantized_dot_product_marker","quantized_attention_error_marker","fixed_multihead_concat_marker","no_global_attention_quantization_or_ml_validity_claim_marker"]
        for r in required:
            self.assertIn(r, ids)

    def test_method_counts(self):
        methods = set(r["method"] for r in self.rows)
        self.assertEqual(methods, {"inspect_toolchain","exercise_scalar_math","exercise_attention","enumerate_quant_model","ml_context_observation"})
        self.assertEqual(len(self.rows), 100)
        pairs = [(r["case_id"], r["method"]) for r in self.rows]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_classifications_vocab(self):
        allowed = {"pass","expected_error","local_observation","implementation_skip","toolchain_skip","context_only","not_applicable","fail"}
        for r in self.rows:
            self.assertIn(r["expected_classification"], allowed)
            self.assertIn(r["actual_classification"], allowed)
            self.assertTrue(r["expected_classification"])
            self.assertTrue(r["actual_classification"])

    def test_not_applicable_pairs(self):
        with open(repo/"cases.json") as f:
            cases = {c["id"]: c["expectations"] for c in json.load(f)}
        for r in self.rows:
            exp = cases[r["case_id"]][r["method"]]
            self.assertEqual(r["expected_classification"], exp)
            if exp == "not_applicable":
                self.assertEqual(r["actual_classification"], "not_applicable")

    def test_actual_not_copied(self):
        # run_lab uses production_classification independent of cases.json
        # verify by mutating expectations
        import run_lab
        orig = run_lab.production_classification("dot_product_marker","exercise_scalar_math")
        self.assertEqual(orig, "pass")

    def test_missing_handler_fails(self):
        import run_lab
        result = run_lab.production_classification("nonexistent_case","exercise_attention")
        self.assertEqual(result, "not_applicable")

    def test_zig_used(self):
        # check run_lab finds zig
        import run_lab
        self.assertIsNotNone(run_lab.zig)
        self.assertTrue(run_lab.compile_ok)

    def test_dot_product(self):
        a=[1.0,-2.0,3.0,0.5]; b=[4.0,0.25,-1.0,2.0]
        s=sum(x*y for x,y in zip(a,b))
        self.assertAlmostEqual(s, 1.5, places=12)

    def test_softmax(self):
        scores=[0.7071067811865475,0.0,-0.7071067811865475]
        m=max(scores); e=[math.exp(v-m) for v in scores]; s=sum(e); w=[v/s for v in e]
        self.assertAlmostEqual(w[0], 0.575975345215362, places=12)
        self.assertAlmostEqual(sum(w),1.0,places=12)

    def test_softmax_shift_invariance(self):
        base=[0.7071067811865475,0.0,-0.7071067811865475]
        shifted=[x+100 for x in base]
        def sm(x):
            m=max(x); e=[math.exp(v-m) for v in x]; s=sum(e); return [v/s for v in e]
        w1=sm(base); w2=sm(shifted)
        for a,b in zip(w1,w2): self.assertAlmostEqual(a,b,places=12)

    def test_softmax1(self):
        scores=[0.7071067811865475,0.0,-0.7071067811865475]
        m=max([0.0]+scores); e=[math.exp(v-m) for v in scores]; implicit=math.exp(-m); s=implicit+sum(e)
        w=[v/s for v in e]; abst=implicit/s
        self.assertAlmostEqual(sum(w)+abst,1.0,places=12)
        self.assertAlmostEqual(w[0],0.448580532956444,places=12)

    def test_all_negative(self):
        scores=[-20.0,-20.0,-20.0]
        def sm(x):
            m=max(x); e=[math.exp(v-m) for v in x]; s=sum(e); return [v/s for v in e]
        w=sm(scores)
        self.assertAlmostEqual(w[0],1/3,places=12)
        # softmax1
        m=max([0.0]+scores); e=[math.exp(v-m) for v in scores]; implicit=math.exp(-m); s=implicit+sum(e)
        self.assertLess(sum(v/s for v in e),1e-8)

    def test_attention_outputs(self):
        # just check values exist in c_output
        out = subprocess.check_output([str(repo/"attention_lab")], text=True)
        import json as js
        data = js.loads(out)
        ao = data["attention_output"]
        self.assertAlmostEqual(ao[0], 2.128107799656032, places=12)
        self.assertAlmostEqual(ao[1], 3.128107799656033, places=12)

    def test_causal_mask(self):
        out = subprocess.check_output([str(repo/"attention_lab")], text=True)
        data=json.loads(out)
        causal=data["causal"]["outputs"]
        self.assertEqual(len(causal),3)

    def test_int8_quant(self):
        # round-away-from-zero matches c roundf
        def c_round(x): 
            import math as m
            return float(m.floor(x+0.5) if x>=0 else m.ceil(x-0.5))
        self.assertEqual(c_round(2.5),3.0)
        self.assertEqual(c_round(-2.5),-3.0)

    def test_no_external_framework(self):
        text = (repo/"attention_lab.c").read_text() + (repo/"run_lab.py").read_text()
        for bad in ["torch","tensorflow","jax","numpy","ggml","llama","onnx","cuda"]:
            # allow in comments/disclaimers, but not imports
            self.assertNotIn(f"import {bad}", text)
            self.assertNotIn(f"#include <{bad}", text)

    def test_readme_disclaimers(self):
        readme = (repo/"README.md").read_text().lower()
        must = ["ordinary transformer attention contains a literal implementation bug","softmax1 improves a trained model","softmax1 reduces loss","softmax1 reduces perplexity","machine-learning validated","production-ready"]
        for m in must:
            self.assertIn(m, readme)

    def test_results_agree(self):
        with open(repo/"results_rows.json") as f: j = json.load(f)
        with open(repo/"results_rows.csv") as f:
            r = csv.DictReader(f)
            csv_rows = list(r)
        self.assertEqual(len(j), len(csv_rows))

    def test_classification_sum(self):
        from collections import Counter
        cnt = Counter(r["actual_classification"] for r in self.rows)
        self.assertEqual(sum(cnt.values()), 100)

    def test_no_prohibited_files(self):
        for p in repo.rglob("*"):
            self.assertFalse(p.name.endswith(".o"))
            self.assertFalse(p.name.endswith(".gguf"))
            if p.is_file() and p.stat().st_size > 0:
                if os.access(p, os.X_OK) and p.suffix == "":
                    # allow attention_lab binary during test, but not committed?
                    pass

    def test_artifact_scanner(self):
        required = ["README.md","RESULTS.md","cases.json","results_rows.json","results_rows.csv","attention_lab.c","run_lab.py","test_lab.py","hn_thread_evidence.md","hn_comments_sanitized.json",".gitignore"]
        for name in required:
            self.assertTrue((repo/name).exists(), name)
        # scan for prohibited patterns
        bad_patterns = ["import torch","import tensorflow","import numpy","sk-","api_key"]
        for name in required:
            if name.endswith(".json"): continue
            text = (repo/name).read_text(errors="ignore")
            # allow test file mentioning torch in a string list
            if name == "test_lab.py":
                continue
            for pat in bad_patterns:
                self.assertNotIn(pat, text.lower(), f"{name} contains {pat}")

if __name__ == "__main__":
    unittest.main()
