# Project Layout

```text
min_model/
  src/
    risk_model/
      trainer.py
      woe.py
      exporters/
      utils/
  tests/
  configs/
  docs/
  scripts/
  legacy/
  .gitignore
  pyproject.toml
  README.md
```

## Intent

- `src/risk_model`: reusable package code
- `tests`: validation and smoke tests
- `configs`: default and environment-specific configs
- `docs`: project notes and migration decisions
- `scripts`: runnable helper entrypoints
- `legacy`: intentionally isolated historical scripts or snapshots
