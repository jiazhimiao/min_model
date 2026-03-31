---
name: risk-feature-binning-analyst
description: Analyze large-scale risk features and score outputs for binning, monotonic ordering, delinquency lift, sample share, and rule readiness. Use when Codex needs to support a risk strategy analyst reviewing 1700+ features, designing manual or semi-manual bins, checking overdue-rate trends, selecting candidate thresholds for launch, or performing recurring MA score monitoring for ordering and separability.
---

# Risk Feature Binning Analyst

## Overview

Use this skill to support recurring risk strategy analysis on high-dimensional feature sets and MA-prefixed scores. Focus on business-usable binning, monotonicity after binning, overdue-rate and sample-share interpretation, and turning stable findings into candidate online rules with thresholds.

Prefer pragmatic analysis over rigid formulas. Do not assume equal-width bins are appropriate. Let the observed data distribution, sample sufficiency, business meaning, and monotonic ordering determine the final grouping.

Default settlement-status normalization:

- `0`: unpaid
- `1`: early_settlement
- `2`: same_day_settlement
- `3`: extension_settlement_before_overdue
- `4`: extension_settlement_after_overdue, merge into `2`
- `5`: merge into `3`

When the source data contains `settlestatus`, treat the normalized analysis status as:

- `4 -> 2`
- `5 -> 3`

Use this normalized status consistently in analysis and reporting unless the user explicitly asks for the raw-status view.

## Core Workflow

### 1. Clarify the analysis target

Identify which of these tasks the user needs:

- Feature screening across many variables
- Single-feature deep dive and manual binning
- Candidate threshold or rule design
- Weekly MA score monitoring
- Review write-up for launch or rejection

Confirm the target label, observation window, and whether the main goal is interpretability, launch readiness, or weekly health monitoring.

If `settlestatus` exists in the source data, normalize it first and mention that the review follows the default merged-status convention.

### 2. Classify the variable before binning

Separate the object into one of these buckets:

- Continuous or high-cardinality numeric feature
- Low-cardinality numeric feature
- Categorical or code-like feature
- MA-prefixed score

Use this classification to decide how aggressive the grouping should be.

### 3. Build bins with monotonicity as a hard requirement

Start from the data, not from a fixed formula.

Apply these heuristics:

- If distinct values are small, inspect natural values first and keep business-readable cut points when possible.
- If distinct values exceed about 20, assume binning is needed, but do not default to equal-width cuts.
- Use quantiles, natural breaks, sparse-tail merging, or manual thresholds as starting points only.
- Merge adjacent bins when sample share is too small, overdue rate is unstable, or monotonicity breaks because of noise.
- Allow partial grouping for small-value regions when those values carry business meaning and still preserve order.
- Keep missing values, zeros, special codes, or sentinel values separate when they may signal a distinct risk population.

Reject a binning proposal if it looks statistically neat but cannot be explained or used in a rule.

### 4. Evaluate each binned feature

For each final binning result, check at minimum:

- Sample count and sample share by bin
- Overdue rate or bad rate by bin
- Direction and monotonicity of the target trend after binning
- Whether jumps between adjacent bins are meaningful or mostly noise
- Whether the cut points are simple enough for implementation

Treat monotonicity as more important than over-optimizing the number of bins. Prefer fewer, interpretable, stable bins over many fragile bins.

### 5. Turn useful features into candidate rules

When a feature looks usable, propose rules in analyst language rather than only presenting a chart.

Include:

- The recommended threshold or grouped range
- The expected risk direction
- Which population would be hit
- Why the threshold is operationally reasonable
- Any concerns about sample size, stability, or edge cases

If the feature is informative but not clean enough for launch, say so directly and recommend observation or re-binning instead of forcing a rule.

### 6. Monitor MA scores weekly

For MA-prefixed scores, run a weekly inspection focused on:

- Ordering monotonicity after binning
- Separation between bins, especially between adjacent middle bins
- Population concentration or score compression
- Whether the highest-risk and lowest-risk bins remain distinguishable
- Whether drift or instability suggests recalibration or investigation

A score can still be problematic even when headline lift looks acceptable. Call out cases where ranking weakens, bins overlap too much, or monotonicity breaks.

## Decision Rules

Use these defaults unless the user provides a better standard:

- Prefer manual or semi-manual bins over purely mechanical bins when business interpretation matters.
- Keep bin counts moderate; too many bins usually reduce stability and explainability.
- Separate missing or special values when they likely represent a different behavioral group.
- Do not force full monotonicity if the only way to achieve it is to create absurd or non-implementable thresholds; instead propose a simpler regrouping.
- If a feature has weak value but one sub-range is clearly risky and stable, allow a partial-bin rule recommendation.

## Output Style

When reporting results, structure the response so a strategy analyst can act on it quickly.

Include:

- What was analyzed
- How the bins were chosen or adjusted
- Whether monotonicity passed
- What the overdue-rate and sample-share pattern means
- Whether the feature or score is fit for launch, monitoring, or rejection
- Concrete next action

Use concise business wording such as:

- "Recommend launch candidate"
- "Observe only, not ready for rule"
- "Re-bin and retest"
- "Ordering weakened this week"
- "Middle bins overlap; separability is insufficient"

## Reference Material

Read [references/binning-review-checklist.md](references/binning-review-checklist.md) when you need a reusable checklist for feature review, MA score weekly inspection, or concise launch recommendations.

Read [references/weekly-report-template.md](references/weekly-report-template.md) when you need to turn the review result into a recurring weekly summary.

## Bundled Scripts

Use [scripts/review_feature_bins.py](scripts/review_feature_bins.py) to batch-review many features and export:

- `feature_summary.csv`
- `bin_details.csv`

Example:

```bash
python .codex/skills/risk-feature-binning-analyst/scripts/review_feature_bins.py data.pkl --target y --exclude-prefix MA --output-dir output/binning_review
```

Use [scripts/weekly_ma_check.py](scripts/weekly_ma_check.py) to review `MA` scores and export:

- `ma_score_summary.csv`
- `ma_score_bin_details.csv`
- `ma_weekly_review.md`

Example:

```bash
python .codex/skills/risk-feature-binning-analyst/scripts/weekly_ma_check.py data.pkl --target y --prefix MA --output-dir output/ma_weekly_check
```

Use [scripts/export_score_feature_template.py](scripts/export_score_feature_template.py) to fill the Excel template [score-band-improvement-review-template.xlsx](/D:/Trae_pro/min_model/score-band-improvement-review-template.xlsx). The current template has separate `评分` and `特征` sheets, and the script can populate either or both. It automatically normalizes `settlestatus` using `4 -> 2` and `5 -> 3`.

Example:

```bash
python .codex/skills/risk-feature-binning-analyst/scripts/export_score_feature_template.py data.pkl --score-column MA_demo --feature-column LOANCOUNT --output-path output/score_band_review.xlsx
```

Treat these scripts as starting points. If the user's target column, score prefix, or reporting threshold differs, adjust the arguments instead of rewriting the whole workflow.
