# Test Result

Run target:

- `python tests/test_model_trainer.py`

Run location:

- `D:\Trae_pro\model_\min_model`

Dataset:

- `data.pkl`
- Rows: `11452`
- Columns: `1763`
- Target distribution: `0=8286`, `1=3166`

Final status:

- `logistic`: success
- `lightgbm`: success
- `xgboost`: success

Metrics summary:

| model | test_auc | test_ks | oot_auc | oot_ks | var_count |
|---|---:|---:|---:|---:|---:|
| logistic | 0.6329 | 0.2069 | 0.6551 | 0.2418 | 10 |
| lightgbm | 0.6736 | 0.2747 | 0.6672 | 0.2603 | 50 |
| xgboost | 0.6683 | 0.2756 | 0.6896 | 0.3022 | 50 |

Generated outputs:

- `output/lr_logistic_model.pkl`
- `output/lr_scorecard.xlsx`
- `output/lr_woe_binner.pkl`
- `output/lgb_lightgbm_model.pkl`
- `output/lgb_lightgbm_model.pmml`
- `output/xgb_xgboost_model.pkl`
- `output/xgb_xgboost_model.pmml`

Notes:

- `trainer.py` now uses the extracted tool-based `WOEBinning` implementation from `src/risk_model/woe.py`.
- The legacy in-file `WOEBinning` definition has been removed from `trainer.py`.
