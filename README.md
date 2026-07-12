# Meal Demand Forecasting

A system for forecasting daily meal demand in IIITD's mess.

## Goal

To minimize food wastage by predicting number of students for each meal.

## Team

- Shaurya Pratap Singh
- Yash Kumar

## Status

Feature Set V1 is complete for all four meals: cleaned, engineered, tuned,
ensembled, and validated on a real 2026 holdout. The four meals show
meaningfully different behavior (different default-winning models,
different feature importance shapes, different tuning gains, see
`results.md` §4), early evidence that V2 may need to branch per meal
rather than apply one shared change everywhere.

Next step is error analysis on the time series cross validation folds (not the 2026
holdout) to confirm this and decide what each meal's Feature Set V2 should
contain.
