# Cleanup Scope

This folder was created as a minimal project snapshot derived from the larger repository.

## Kept

- The unified training framework centered on `model_trainer.py`
- A lightweight package layout under `src/risk_model`
- A copied integration-style test entry for quick validation

## Not copied

- Data files (`*.pkl`, `*.csv`, `*.xlsx`)
- Model artifacts (`*.pmml`, `*.h5`, exported reports, plots)
- Runtime logs and cache files
- Historical project snapshots and duplicated output directories
- Deep learning scripts that depend on hard-coded paths such as `D:\...` or `/root/...`

## Suggested next step

If you want to evolve this folder further, the next best move is to split `trainer.py` into:

1. `woe.py`
2. `feature_selection.py`
3. `exporters.py`
4. `trainer.py`

That would make the code easier to test and maintain without changing the original repository.
