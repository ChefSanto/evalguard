## Quickstart

    from evalguard import EvalHarness

    h = EvalHarness.from_run_config("runs/exp3_s7/run_config.json")  # guards 1, 2
    h.register_model(predict_fn, handles_chaos=True)                # guard 3
    report = h.run(batches, metrics={"mae": mae_fn}, chaos_values=chaos)
    report.assert_parity(tol=1e-3)                                  # guard 4

    from evalguard import collect
    collect("results/grid_manifest.json", "results/eval")           # guard 5

## Training-side integration (two lines after final-epoch eval)

    from evalguard import RunConfig
    RunConfig(dataset=..., epochs=..., strategy=..., seed=...,
              chaos_source=..., gate_low=..., gate_high=...,
              text_w_raw=..., audio_w_raw=..., visual_w_raw=...,
              in_training_metrics={"mae": float(final_mae)}).dump(run_dir / "run_config.json")

## Boundaries (explicit non-goals)

- Does not touch training (LR schedules, checkpointing).
- Does not choose or validate metrics; you pass metric callables - evalguard
  guarantees the conditions under which they are computed.
- Does not do experiment tracking; one JSON report per run, on disk, diffable.
- Framework-agnostic: any callable works. Adapters would be downstream plugins.

## Exhibits (from the project's own archive, found by gate_instability #3)

- 38 params tags with no segments in the pooled training log, and 19 log
  segments with no tags - the archive is inconsistent in both directions
  (guard 5's bug class, at scale, discovered after the fact).
- Two runs (mosi/mosei 40ep) showing the guard-3 failure mode in stored
  data: dormant in training, oracle-chaos-activated at final eval.

## Status

v0.1.0 - single-file library (`evalguard.py`); can be split into a package
later without API changes. `RunConfig` fields mirror the Paper-1 run grid
(`gate_params_all_runs.csv` columns). Not yet exercised against the real
training script; first live use is planned for the CH-SIMS Stage-1 runs.

## Quickstart

    from evalguard import EvalHarness

    h = EvalHarness.from_run_config("runs/exp3_s7/run_config.json")  # guards 1, 2
    h.register_model(predict_fn, handles_chaos=True)                # guard 3
    report = h.run(batches, metrics={"mae": mae_fn}, chaos_values=chaos)
    report.assert_parity(tol=1e-3)                                  # guard 4

    from evalguard import collect
    collect("results/grid_manifest.json", "results/eval")           # guard 5

## Training-side integration (two lines after final-epoch eval)

    from evalguard import RunConfig
    RunConfig(dataset=..., epochs=..., strategy=..., seed=...,
              chaos_source=..., gate_low=..., gate_high=...,
              text_w_raw=..., audio_w_raw=..., visual_w_raw=...,
              in_training_metrics={"mae": float(final_mae)}).dump(run_dir / "run_config.json")

## Boundaries (explicit non-goals)

- Does not touch training (LR schedules, checkpointing).
- Does not choose or validate metrics; you pass metric callables - evalguard
  guarantees the conditions under which they are computed.
- Does not do experiment tracking; one JSON report per run, on disk, diffable.
- Framework-agnostic: any callable works. Adapters would be downstream plugins.

## Exhibits (from the project's own archive, found by gate_instability #3)

- 38 params tags with no segments in the pooled training log, and 19 log
  segments with no tags - the archive is inconsistent in both directions
  (guard 5's bug class, at scale, discovered after the fact).
- Two runs (mosi/mosei 40ep) showing the guard-3 failure mode in stored
  data: dormant in training, oracle-chaos-activated at final eval.

## Status

v0.1.0 - single-file library (`evalguard.py`); can be split into a package
later without API changes. `RunConfig` fields mirror the Paper-1 run grid
(`gate_params_all_runs.csv` columns). Not yet exercised against the real
training script; first live use is planned for the CH-SIMS Stage-1 runs.
