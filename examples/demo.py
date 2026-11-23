"""Five-minute demo: happy path, then evalguard catching all 5 original bug
classes. Run:  python3 examples/demo.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalguard import (
    EvalHarness, RunConfig, collect,
    ConfigError, ChaosGuardError, ParityError, ManifestError,
)

TRAIN_MAE, DRIFT_MAE = 0.8792, 1.00923   # the real incident numbers


def mae(t, p):
    s, n = 0.0, 0
    for a, b in zip(t, p):
        s += abs(a - b)
        n += 1
    return s / max(n, 1)


def show(title):
    print()
    print(f"=== {title} ===")


def caught(exc):
    print(f"  CAUGHT -> {type(exc).__name__}: {str(exc).splitlines()[0]}")


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    cfg = RunConfig(dataset="mosi", epochs=100, strategy="zeros", seed=7,
                    chaos_source="zeros", gate_low=0.236, gate_high=1.50,
                    text_w_raw=0.998, audio_w_raw=0.0008, visual_w_raw=0.0008,
                    in_training_metrics={"mae": TRAIN_MAE})
    rc = tmp / "run_config.json"
    cfg.dump(rc)
    show("HAPPY PATH: eval that matches training passes all guards")
    h = EvalHarness.from_run_config(rc, out_dir=tmp / "out")
    h.register_model(lambda x, chaos_score: x + TRAIN_MAE, handles_chaos=True)
    rep = h.run([(float(i), float(i)) for i in range(8)], {"mae": mae})
    rep.assert_parity(tol=1e-3)
    print(f"  parity OK; report at {rep.path}")

    show("BUG CLASS 1: run_config.json missing -> refused, no invented defaults")
    try:
        EvalHarness.from_run_config(tmp / "no_such_config.json")
    except FileNotFoundError as e:
        caught(e)

    show("BUG CLASS 2: post-hoc --chaos_source override -> fingerprint mismatch")
    d = json.loads(rc.read_text())
    d["chaos_source"] = "detector"
    rc.write_text(json.dumps(d))
    try:
        EvalHarness.from_run_config(rc)
    except ConfigError as e:
        caught(e)
    cfg.dump(rc)   # restore

    show("BUG CLASS 3: chaos_score=None with active gate -> no silent dormant path")
    h = EvalHarness.from_run_config(rc, out_dir=tmp / "out2")
    h.register_model(lambda x, chaos_score: x, handles_chaos=False)
    try:
        h.run([(1.0, 1.0)] * 4, {"mae": mae}, chaos_values=[0.5, None, 0.5, None])
    except ChaosGuardError as e:
        caught(e)

    show(f"BUG CLASS 4: post-hoc metric drift {DRIFT_MAE} vs {TRAIN_MAE} -> quarantined")
    h = EvalHarness.from_run_config(rc, out_dir=tmp / "out3")
    h.register_model(lambda x, chaos_score: x + DRIFT_MAE, handles_chaos=True)
    rep = h.run([(float(i), float(i)) for i in range(8)], {"mae": mae})
    try:
        rep.assert_parity(tol=1e-3)
    except ParityError as e:
        caught(e)
    print(f"  quarantined report: {(tmp / 'out3' / '_QUARANTINED' / 'report.json').exists()}")
    show("BUG CLASS 5: packaging grid with a missing eval JSON -> named loudly")
    (tmp / "eval_a.json").write_text("{}")
    (tmp / "grid.json").write_text(json.dumps([{"file": "eval_a.json"},
                                               {"file": "eval_b.json"}]))
    try:
        collect(tmp / "grid.json", tmp)
    except ManifestError as e:
        caught(e)

    print()
    print("All five original bug classes are caught before they cost anyone weeks.")
