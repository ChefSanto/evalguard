"""evalguard - eval-harness hardening kit for multimodal runs (A2H2 artifact #2).

Wraps the EVAL LOOP (not the model): callable in, validated report out.
Guards, in order of the original incident:
  1. single-source config   - eval must load the training-written run_config.json;
     missing file or missing field -> hard error, never a silent default.
  2. config-drift detection - every config carries a fingerprint over its
     fields; post-hoc mutation (e.g. a --chaos_source override) is caught
     BEFORE any compute.
  3. chaos null-guard       - chaos_score=None with an active gate raises,
     unless an explicit allow_dormant_fallback=True is passed (and logged).
     The model must declare handles_chaos; no silent capability assumptions.
  4. eval-parity assertion  - post-hoc metrics must match in-training
     final-epoch metrics within tol, else the report is QUARANTINED.
  5. manifest collector     - packaging/aggregation fails hard on file-count
     mismatch vs the declared run grid, naming the missing cells.

Usage:
    from evalguard import EvalHarness, collect
    h = EvalHarness.from_run_config("runs/exp3_s7/run_config.json")
    h.register_model(predict_fn, handles_chaos=True)
    report = h.run(batches, metrics={"mae": mae_fn}, chaos_values=chaos)
    report.assert_parity(tol=1e-3)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

__version__ = "0.1.0"


class EvalGuardError(Exception):
    """Base class for all evalguard failures. All failures are loud."""


class ConfigError(EvalGuardError):
    pass


class ChaosGuardError(EvalGuardError):
    pass


class ParityError(EvalGuardError):
    pass


class ManifestError(EvalGuardError):
    pass


REQUIRED_FIELDS = (
    "dataset", "epochs", "strategy", "seed", "chaos_source",
    "gate_low", "gate_high",
    "text_w_raw", "audio_w_raw", "visual_w_raw",
)
GATE_ACTIVE_STRATEGIES = ("zeros", "detector", "frozentpl")


def config_fingerprint(d: Dict[str, Any]) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunConfig:
    dataset: str
    epochs: int
    strategy: str
    seed: int
    chaos_source: str
    gate_low: float
    gate_high: float
    text_w_raw: float
    audio_w_raw: float
    visual_w_raw: float
    in_training_metrics: Dict[str, float] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return config_fingerprint({k: getattr(self, k) for k in REQUIRED_FIELDS})

    @classmethod
    def load(cls, path: Any) -> "RunConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"run_config.json not found at {p}. evalguard refuses to "
                f"re-derive config defaults (bug class 1: silent defaults)."
            )
        d = json.loads(p.read_text())
        missing = [k for k in REQUIRED_FIELDS if k not in d]
        if missing:
            raise ConfigError(
                f"run_config.json is missing required fields {missing}; "
                f"refusing to invent defaults (bug class 1)."
            )
        cfg = cls(**{k: d[k] for k in REQUIRED_FIELDS},
                  in_training_metrics=d.get("in_training_metrics", {}))
        declared = d.get("config_fingerprint")
        if declared is None:
            raise ConfigError(
                "run_config.json has no config_fingerprint; cannot verify "
                "single-source config (bug class 2)."
            )
        actual = cfg.fingerprint()
        if actual != declared:
            raise ConfigError(
                f"config fingerprint mismatch: computed {actual} != recorded "
                f"{declared}. The eval config was mutated after training "
                f"(bug class 2: stale/post-hoc config, e.g. a --chaos_source "
                f"override). Refusing to evaluate."
            )
        return cfg

    def dump(self, path: Any) -> None:
        p = Path(path)
        d = {k: getattr(self, k) for k in REQUIRED_FIELDS}
        d["in_training_metrics"] = dict(self.in_training_metrics)
        d["config_fingerprint"] = self.fingerprint()
        p.write_text(json.dumps(d, indent=2, sort_keys=True))


class Report:
    def __init__(self, cfg: RunConfig, results: Dict[str, float], out_dir: Any):
        self.cfg = cfg
        self.results = dict(results)
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / "report.json"
        self.quarantined = False

    def write(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {k: getattr(self.cfg, k) for k in REQUIRED_FIELDS},
            "config_fingerprint": self.cfg.fingerprint(),
            "strategy": self.cfg.strategy,
            "seed": self.cfg.seed,
            "metrics": self.results,
            "quarantined": False,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return self.path

    def _quarantine(self, reason: str) -> None:
        qdir = self.out_dir / "_QUARANTINED"
        qdir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.move(str(self.path), str(qdir / self.path.name))
        (qdir / "QUARANTINE_REASON.txt").write_text(reason + "\n")
        self.quarantined = True

    def assert_parity(self, tol: float = 1e-3) -> bool:
        expected = self.cfg.in_training_metrics
        if not expected:
            raise ParityError(
                "run_config.json records no in_training_metrics; training must "
                "write final-epoch metrics at train time (bug class 4 prevention)."
            )
        drift = []
        for k, exp in expected.items():
            if k not in self.results:
                drift.append(f"{k}: missing from eval metrics")
                continue
            if abs(self.results[k] - exp) > tol:
                drift.append(
                    f"{k}: eval {self.results[k]:.6f} vs in-training {exp:.6f} "
                    f"(|d|={abs(self.results[k] - exp):.2e} > tol {tol:.0e})"
                )
        if drift:
            reason = "Eval-parity assertion failed:\n  " + "\n  ".join(drift)
            self._quarantine(reason)
            raise ParityError(
                reason + f"\nReport quarantined under {self.out_dir / '_QUARANTINED'}"
            )
        return True


Batch = Tuple[Any, Any]


class EvalHarness:
    def __init__(self, cfg: RunConfig, out_dir: Any = "eval_out"):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self._model: Optional[Callable] = None
        self._handles_chaos: Optional[bool] = None
        self.dormant_fallback_used = 0

    @classmethod
    def from_run_config(cls, path: Any, out_dir: Any = "eval_out") -> "EvalHarness":
        return cls(RunConfig.load(path), out_dir=out_dir)

    def register_model(self, fn: Callable, *, handles_chaos: Optional[bool]) -> None:
        if handles_chaos is None:
            raise ChaosGuardError(
                "register_model requires handles_chaos=True or False; evalguard "
                "assumes no model capabilities (bug class 3)."
            )
        self._model = fn
        self._handles_chaos = bool(handles_chaos)

    def run(
        self,
        batches: Iterable[Batch],
        metrics: Dict[str, Callable[[List[Any], List[Any]], float]],
        chaos_values: Optional[Iterable[Any]] = None,
        allow_dormant_fallback: bool = False,
    ) -> Report:
        if self._model is None:
            raise EvalGuardError(
                "call register_model(predict_fn, handles_chaos=...) before run()"
            )
        gate_active = self.cfg.strategy in GATE_ACTIVE_STRATEGIES
        chaos_list: Optional[List[Any]] = list(chaos_values) if chaos_values is not None else None
        if chaos_list is not None and gate_active:
            n_none = sum(1 for c in chaos_list if c is None)
            if n_none and self._handles_chaos is False:
                raise ChaosGuardError(
                    f"{n_none} chaos_score=None values, but the registered model "
                    f"declared handles_chaos=False and the gate is ACTIVE "
                    f"(strategy={self.cfg.strategy}). Continuing would silently "
                    f"take the dormant gate path (bug class 3)."
                )
            if n_none and not allow_dormant_fallback:
                raise ChaosGuardError(
                    f"{n_none} chaos_score=None values with an active gate. The "
                    f"dormant fallback exists only behind "
                    f"allow_dormant_fallback=True, and its use is logged "
                    f"(bug class 3)."
                )
            if n_none and allow_dormant_fallback:
                self.dormant_fallback_used = n_none
                print(
                    f"evalguard: DORMANT FALLBACK used for {n_none} rows "
                    f"(explicit allow_dormant_fallback=True); this is logged",
                    file=sys.stderr,
                )
        preds: List[Any] = []
        tgts: List[Any] = []
        for idx, (x, y) in enumerate(batches):
            c = chaos_list[idx] if chaos_list is not None and idx < len(chaos_list) else None
            preds.append(self._model(x, chaos_score=c))
            tgts.append(y)
        results = {name: fn(tgts, preds) for name, fn in metrics.items()}
        rep = Report(self.cfg, results, self.out_dir)
        rep.write()
        return rep


def collect(manifest_path: Any, runs_dir: Any) -> List[Path]:
    """Verify packaging/aggregation: every declared run file must exist."""
    grid = json.loads(Path(manifest_path).read_text())
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise ManifestError(f"runs directory {runs_dir} does not exist")
    present, missing = [], []
    for cell in grid:
        f = runs_dir / cell["file"]
        (present if f.exists() else missing).append(cell["file"])
    if missing:
        raise ManifestError(
            f"manifest declares {len(grid)} run files; {len(missing)} missing under "
            f"{runs_dir}: " + ", ".join(missing[:10]) +
            (" ..." if len(missing) > 10 else "") + " (bug class 5)"
        )
    return [runs_dir / c["file"] for c in grid]
