# Metric Framework

## KPI Layers

Use KPI layers to avoid mixing symptoms with root causes:

1. Input quality: traffic volume, registration rate, auth completion, anti-fraud hit rate
2. Decision quality: approval rate, manual review rate, offer acceptance
3. Booking quality: disbursement rate, amount mix, tenor mix, price mix
4. Performance quality: FPD, 7+, 30+, roll rate, fraud loss, charge-off
5. Recovery quality: connect rate, promise-to-pay, cure rate, recovery amount
6. Economic quality: CAC, contribution margin, expected loss, ROI

## Recommended Diagnosis Cuts

- Time cut: day, week, policy release window, vintage
- Customer cut: new/repeat, customer tier, score band
- Traffic cut: channel, campaign, creative, merchant, partner
- Product cut: product line, amount band, tenor, price band
- Risk cut: fraud score, credit score, rule-hit label, manual review reason
- Geography and device cut: province, city tier, OS, device brand

## Tradeoff Checklist

Before recommending a policy change, check:

- Will approval move mostly in good customers or bad customers?
- Will delinquency improvement come from true risk reduction or just volume suppression?
- Will manual review or collections workload exceed capacity?
- Will the change distort channel mix or partner incentives?
- Is there any fairness, explainability, or compliance concern?

## Typical Output Template

Use this compact structure in analysis:

1. Problem: which KPI moved and in which segment
2. Diagnosis: top 2-4 likely drivers with supporting evidence
3. Action: exact strategy change with population, threshold, and routing
4. Impact: expected change on approval, risk, workload, and profit
5. Validation: backtest or experiment design and rollback trigger
