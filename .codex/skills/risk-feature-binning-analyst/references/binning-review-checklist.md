# Binning Review Checklist

Use this reference when reviewing feature bins or MA score bins.

## Feature Review

### Default status mapping

When `settlestatus` is present, normalize it before analysis:

- `0`: unpaid
- `1`: early_settlement
- `2`: same_day_settlement
- `3`: extension_settlement_before_overdue
- `4 -> 2`
- `5 -> 3`

Unless the user explicitly asks for raw-status results, all conclusions should follow this merged mapping.

### Scope

- Confirm target definition and overdue window.
- Confirm whether the task is feature screening, threshold design, or launch review.
- Confirm whether missing, zero, or special values need separate treatment.

### Binning

- Inspect distinct-value count before choosing a strategy.
- If distinct values are greater than about 20, start with a grouped approach.
- Do not default to equal-width bins unless the distribution is unusually regular.
- Merge sparse adjacent bins when sample share is too small or bad-rate noise is high.
- Preserve interpretable cut points where possible.

### What to inspect after binning

- Sample count by bin
- Sample share by bin
- Overdue or bad rate by bin
- Monotonic ordering by bin
- Sharp reversals that suggest noise or bad cuts
- Whether the final cut points are simple enough for online implementation

### Launch recommendation

Recommend launch only when most of these are true:

- Post-binning ordering is monotonic or close to monotonic with a clear business explanation
- Sample share is not excessively concentrated in one unstable tail bin
- Risk separation is visible and not driven by tiny samples
- Thresholds are simple enough to implement and explain
- The rule does not depend on fragile micro-bins

Use cautious wording when needed:

- "Useful for observation, not launch"
- "Only the high-risk tail is actionable"
- "Needs regrouping before threshold selection"

## MA Score Weekly Check

### Minimum checks

- Compare ordered bins week over week
- Confirm bad-rate ranking remains monotonic
- Check whether adjacent bins still separate cleanly
- Check whether the score compresses into a narrow band
- Flag sudden share migration across bins

### Common warning signs

- Middle bins overlap heavily in bad rate
- Top bin no longer carries the worst risk
- One bin absorbs too much population and weakens separation
- Ordering only holds after extreme manual adjustment

### Suggested conclusion labels

- "Stable ordering and separation"
- "Ordering stable, separation weakened"
- "Ordering broken, investigate drift"
- "Usable for monitoring only"
