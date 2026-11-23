"""Bug-injection tests: each test recreates one original bug class and asserts
evalguard catches it. This file IS the product demo - run either:
    python3 tests/test_bug_injection.py      (no pytest needed)
    pytest tests/ -v
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalguard import (
    EvalHarness, RunConfig, collect,
    ConfigError, ChaosGuardError, ParityError, ManifestError,
)

BASE = dict(dataset="mosi", epochs=100, strategy="zeros", seed=7,
            chaos_source="zeros", gate_low=0.236, gate_high=1.50,
            text_w_raw=0.998, audio_w_raw=0.0008, visual_w_raw=0.0008)
TRAIN_MAE = 0.8792   # the real in-training value from the incident
DRIFT_MAE = 1.00923  # the real drifted post-hoc value from the incident


def mae(t, p):
    n = 0
    s = 0.0
    for a, b in zip(t, p):
        s += abs(a - b)
        n += 1
    return s / max(n, 1)


def make_run_dir(tmp, metrics=None, **overrides):
    cfg = RunConfig(**{**BASE, **overrides},
                    in_training_metrics=metrics or {"mae": TRAIN_MAE})
    p = Path(tmp) / "run_config.json"
    cfg.dump(p)
    return p


def batches(n=8):
    return [(float(i), float(i)) for i in range(n)]


def test_1_missing_config_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            EvalHarness.from_run_config(Path(tmp) / "run_config.json")
        except FileNotFoundError as e:
            assert "refuses to re-derive" in str(e)
            return
        raise AssertionError("evalguard accepted a missing run_config.json")


def test_1b_missing_field_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)
        d = json.loads(p.read_text())
        del d["chaos_source"]
        p.write_text(json.dumps(d))
        try:
            RunConfig.load(p)
        except ConfigError as e:
            assert "missing required fields" in str(e)
            return
        raise AssertionError("evalguard invented a default for a missing field")


def test_2_mutated_config_detected_before_compute():
    """The original bug: post-hoc eval overrode --chaos_source to 'detector'."""
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)
        d = json.loads(p.read_text())
        d["chaos_source"] = "detector"   # post-hoc mutation, fingerprint stale
        p.write_text(json.dumps(d))
        try:
            EvalHarness.from_run_config(p)
        except ConfigError as e:
            assert "fingerprint mismatch" in str(e)
            return
        raise AssertionError("evalguard evaluated with a mutated config")


def test_3a_chaos_none_with_incapable_model_raises():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)   # strategy zeros -> gate active
        h = EvalHarness.from_run_config(p, out_dir=Path(tmp) / "out")
        h.register_model(lambda x, chaos_score: 0.0, handles_chaos=False)
        try:
            h.run(batches(), {"mae": mae}, chaos_values=[0.5, None, 0.5, None])
        except ChaosGuardError as e:
            assert "dormant gate path" in str(e)
            return
        raise AssertionError("silent dormant path was allowed")


def test_3b_capability_declaration_required():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)
        h = EvalHarness.from_run_config(p, out_dir=Path(tmp) / "out")
        try:
            h.register_model(lambda x, chaos_score: 0.0, handles_chaos=None)
        except ChaosGuardError:
            return
        raise AssertionError("register_model accepted an undeclared capability")


def test_3c_dormant_fallback_requires_explicit_flag():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)
        h = EvalHarness.from_run_config(p, out_dir=Path(tmp) / "out")
        h.register_model(lambda x, chaos_score: 0.0, handles_chaos=True)
        try:
            h.run(batches(), {"mae": mae}, chaos_values=[0.5, None])
        except ChaosGuardError:
            pass
        else:
            raise AssertionError("dormant fallback ran without the explicit flag")
        rep = h.run(batches(), {"mae": mae}, chaos_values=[0.5, None],
                    allow_dormant_fallback=True)
        assert rep.path.exists() and h.dormant_fallback_used == 1


def test_4_parity_drift_is_quarantined():
    """The incident numbers: post-hoc text_corrupt MAE 1.00923 vs 0.8792."""
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)
        h = EvalHarness.from_run_config(p, out_dir=Path(tmp) / "out")
        h.register_model(lambda x, chaos_score: x + DRIFT_MAE, handles_chaos=True)
        rep = h.run(batches(), {"mae": mae})   # every pred off by DRIFT_MAE
        assert rep.path.exists()
        try:
            rep.assert_parity(tol=1e-3)
        except ParityError as e:
            assert "quarantined" in str(e).lower()
            assert not rep.path.exists()                                   # moved
            qdir = Path(tmp) / "out" / "_QUARANTINED"
            assert (qdir / "report.json").exists()                          # quarantined
            assert "Eval-parity assertion failed" in (qdir / "QUARANTINE_REASON.txt").read_text()
            return
        raise AssertionError("14.8% metric drift passed parity")


def test_4b_parity_pass_when_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_run_dir(tmp)
        h = EvalHarness.from_run_config(p, out_dir=Path(tmp) / "out")
        h.register_model(lambda x, chaos_score: x + TRAIN_MAE, handles_chaos=True)
        rep = h.run(batches(), {"mae": mae})
        assert rep.assert_parity(tol=1e-3) is True
        assert not (Path(tmp) / "out" / "_QUARANTINED").exists()


def test_5_manifest_collector_fails_loud():
    """The Nov-13 incident: packaging silently found 0 eval JSONs."""
    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp)
        (rd / "eval_mosi_100ep_zeros_seed7.json").write_text("{}")
        (rd / "eval_mosi_100ep_zeros_seed123.json").write_text("{}")
        grid = [{"file": f} for f in (
            "eval_mosi_100ep_zeros_seed7.json",
            "eval_mosi_100ep_zeros_seed123.json",
            "eval_mosi_100ep_detector_seed7.json")]
        mp = rd / "grid_manifest.json"
        mp.write_text(json.dumps(grid))
        try:
            collect(mp, rd)
        except ManifestError as e:
            assert "eval_mosi_100ep_detector_seed7.json" in str(e)
            return
        raise AssertionError("packaging gap passed silently")


def test_5b_manifest_complete_grid_passes():
    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp)
        files = ["eval_a.json", "eval_b.json"]
        for f in files:
            (rd / f).write_text("{}")
        mp = rd / "grid_manifest.json"
        mp.write_text(json.dumps([{"file": f} for f in files]))
        out = collect(mp, rd)
        assert len(out) == 2


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"{len(ALL) - failed}/{len(ALL)} bug-injection tests passed")
    sys.exit(1 if failed else 0)
