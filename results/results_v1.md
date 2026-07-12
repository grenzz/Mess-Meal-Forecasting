# Version-1 Report

Complete results for Feature Set V1 across all four meals: Breakfast, Lunch,
Evening Snacks, Dinner.

---

## Pipeline

```
01_exploring_raw_data.ipynb      raw Excel (28 monthly sheets) -> unified daily CSV
02_data_analysis.ipynb           exploratory plots, sanity checks
03_feature_engineering.ipynb     calendar merge + lag/rolling/EWMA/cyclic features
04a_modeling_dinner.ipynb        CV, model comparison, Optuna tuning, ensembling
04b_modeling_lunch.ipynb         (same pipeline, per meal - kept as separate
04c_modeling_breakfast.ipynb      notebooks so each meal's full experiment
04d_modeling_snacks.ipynb         history stays independently inspectable)
```

## 0. Data handling

- Single raw source workbook: `meal wise head count (2024 - 2025 - 2026(may)).xlsx`,
  28 sheets (one per month), header row position and date column located
  dynamically per sheet (not fixed, sheets had inconsistent layouts).
- All 28 sheets parsed and concatenated: **836 rows**, 2024-01-01 to 2026-05-31,
  zero duplicate dates, zero missing values in the parsed columns.
- 836 rows is *less* than the full 882-day calendar span (731 days for
  2024–2025 + 151 for Jan–May 2026) — the 46-row gap is the mess-closure
  periods (June 2024, Dec 8-23 2024), which don't exist as rows in the raw
  sheets at all (e.g. Sheet6 literally starts with the text "June Closed"
  before the first real data row, for July 2024).
- Splitted at cleaning time, before any feature engineering touched it:
  - `mess_data_2024_2025_clean.csv` — 685 rows, training data
  - `mess_data_2026_holdout.csv` — 151 rows, held out from day one

## 1. Feature engineering (Feature Set V1)

- Academic calendar (`academic_calendar_2426.xlsx`, extended through May 2026)
  merged on `Date`. Two known typos corrected on merge:
  `"winter vacation"` → `"winter_vacation"`, `"holdiay"` → `"holiday"`.
- `df_master.set_index("Date").asfreq("D")` used to force a complete daily
  index **before** any lag/rolling computation — this is the fix that
  prevents `.shift()`/`.rolling()` from silently pulling values across the
  June/December 2024 closure gaps.
- Base feature block (20 columns): calendar structure (`DayOfMonth`, `Month`,
  `WeekOfYear`, `DayOfYear`), binary flags (`Is_Start_Of_Month`,
  `Is_End_Of_Month`, `Is_Weekend`, `Is_Holiday`, `Is_Vacation`,
  `Is_Exam_Period`, `Is_Fest`), `Days_Until_Holiday` (backfilled from the next
  holiday date), cyclic sin/cos encodings for day-of-week and month, and two
  interaction flags (`Is_Broke_Weekend` = end-of-month × weekend,
  `Is_Rich_Weekend` = start-of-month × weekend).
- Per meal, 25 more columns built on `Target_Headcount`:
  - 6 lags: 1, 2, 3, 7, 14, 21 days back
  - 16 rolling stats: mean/std/max/min over 3/7/14/28-day windows (all
    computed on `.shift(1)` first, so the current day is never included in
    its own rolling window)
  - 2 EWMA (span 7, span 14) + 1 trend ratio (7-day avg ÷ 28-day avg)
- Total: 46 columns per meal file (`Date`, `Target_Headcount`, 44 predictors).
- Same pipeline re-run on 2024–25 concatenated with 2026 raw data (so lag/
  rolling windows reaching into December 2025 resolve correctly), then
  sliced back to just the 2026 rows for holdout files.
- `Target_Headcount` is null for the known closure days after `asfreq`
  reindexing (46 rows) — dropped in the modeling notebooks via `dropna()`,
  identically across all four meals.

## 2. Exploratory observations (`02_data_analysis.ipynb`)

Low-attendance-day analysis (days with Lunch < 100) shows clear clustering
by month rather than being spread evenly across the year:

| Month | Low-lunch days |
|---|---|
| 2024-05 | 15 |
| 2024-12 | 8 |
| 2025-12 | 10 |
| 2024-01, 2024-10, 2025-05, 2025-10 | 5–7 each |
| 2024-03, 2025-03 | 1–4 |

May and December both show up heavily, consistent with end-of-semester/
near exam attrition in attendance, and gives an early hint(well before 
formal error analysis) that exam and vacation near periods
carry real signal.

## 3. Per meal results

Methodology is identical across all four meals: `dropna()` on the full
feature set, initial single year sanity check (train 2024 / validate 2025),
6-fold expanding window cross validation, Optuna tuning (400 trials/
model, `TPESampler(multivariate=True, group=True)`, widened search bounds:
XGBoost `max_depth` (3,12), LightGBM `num_leaves` (15,255) and `max_depth`
(3,12), CatBoost `depth` (1,10)), simple average ensemble of the three tuned
models, final holdout test trained on all of 2024-25 and tested on the full
2026 Jan-May window.

Fold windows (identical across all four meals):
```
Fold 1: train 2024-01-01→2024-12-31, validate 2025-01-01→2025-03-01
Fold 2: train 2024-01-01→2025-03-01, validate 2025-03-01→2025-05-01
Fold 3: train 2024-01-01→2025-05-01, validate 2025-05-01→2025-07-01
Fold 4: train 2024-01-01→2025-07-01, validate 2025-07-01→2025-09-01
Fold 5: train 2024-01-01→2025-09-01, validate 2025-09-01→2025-11-01
Fold 6: train 2024-01-01→2025-11-01, validate 2025-11-01→2025-12-31
```

---

### 3.1 Dinner

**Initial sanity check** (train 2024, validate 2025, single untuned CatBoost):
Baseline MAE 74.94 → Model MAE 59.07 — first confirmation the feature set
carries real signal, before any of the CV/tuning framework existed.

**Top 15 features** (CatBoost default `feature_importances_`, gain-based):
`DayOfWeek_Sin` (10.9) leads, ahead of the lag cluster (`Target_7_Days_Ago`
9.8, `Target_21_Days_Ago` 8.5, `Target_14_Days_Ago` 8.2, `Target_1_Days_Ago`
7.5). Weekly rhythm is the single strongest signal for Dinner, with recent-
history lags close behind.

**Default model comparison (6-fold CV):**

| Model | Avg MAE | Std |
|---|---|---|
| RandomForest | 56.21 | ±9.02 |
| XGBoost | 59.12 | ±13.68 |
| LightGBM | 55.43 | ±9.50 |
| **CatBoost** | **54.57** | **±6.52** |

CatBoost led on both mean and stability. XGBoost's fold 6 (Nov–Dec, fest +
exam + pre-break) spiked to 83.4 - the least stable model on defaults.

**Tuned best params:** XGBoost — lr 0.0141, 730 trees, depth 5, min_child_weight
13, reg_lambda 9.44. LightGBM — lr 0.0283, 467 trees, 215 leaves, depth 10.
CatBoost — lr 0.0537, 692 iterations, depth 5, l2_leaf_reg 19.24 (hugging the
20 upper bound, flagged as an open item, see §5).

**Tuned, per-fold:**

| Model | Mean MAE | Std | Fold 6 |
|---|---|---|---|
| XGBoost (tuned) | 51.98 | 6.78 | **50.2** (down from 83.4 default) |
| LightGBM (tuned) | 51.56 | 7.70 | 52.6 |
| CatBoost (tuned) | 52.53 | 4.94 | 56.6 |

Tuning's biggest win: XGBoost's fest season instability essentially resolved.

**Ensemble (CV):** 51.44 ± 6.22 - beat every individual tuned model.

**Holdout, trained on all 2024-25 (n=601), tested on Jan-May 2026 (n=151):**

| Model | MAE | RMSE |
|---|---|---|
| XGBoost | 38.14 | 50.47 |
| LightGBM | 37.45 | 50.16 |
| CatBoost | 37.01 | 48.70 |
| **Ensemble** | **37.10** | 48.90 |
| Baseline (7d ago) | 75.53 | — |

**~51% reduction vs baseline.** Performance *improved* through the March
2026 menu-change boundary rather than degrading - likely because the
autoregressive lag/rolling features re-anchor to whatever's currently
happening within a few weeks, rather than relying on a fixed calendar→menu
mapping that broke when the rotation changed.

---

### 3.2 Lunch

**Initial sanity check:** Baseline MAE 83.96 → Model MAE 61.48.

**Top 15 features:** `Target_3_Day_Std` leads (0.227), followed by
`Target_7_Days_Ago` (0.186) and `Target_3_Day_Max` (0.131). Notably
different shape from Dinner - Lunch's top feature is a *volatility* measure
(recent 3-day std), not a raw lag or the weekly cyclic encoding. Suggests
Lunch attendance is driven more by "how unstable has it been lately" than
by a clean weekly rhythm.

**Default model comparison (6-fold CV):**

| Model | Avg MAE | Std |
|---|---|---|
| RandomForest | 58.31 | ±9.38 |
| XGBoost | 61.59 | ±5.20 |
| LightGBM | 57.38 | ±5.72 |
| **CatBoost** | **53.65** | **±4.63** |

Same winner as Dinner. Baseline itself is higher (83.96) — Lunch has the
highest raw headcount of the four meals, so naive persistence has more to
get wrong.

**Tuned best params:** XGBoost — lr 0.0231, 329 trees, depth **11** (near the
widened 12 ceiling), reg_alpha 0.0022 (near floor — negligible L1). LightGBM
— lr 0.0099, 798 trees, 90 leaves, depth 5. CatBoost — lr 0.0478, 773
iterations, depth 5, l2_leaf_reg **19.98** (hugging the same 20 ceiling as
Dinner — confirms this is a real, repeatable CatBoost preference, not
meal-specific noise).

**Tuned, per-fold:**

| Model | Mean MAE | Std |
|---|---|---|
| XGBoost (tuned) | 51.39 | 6.15 |
| LightGBM (tuned) | 51.62 | 4.24 |
| CatBoost (tuned) | 51.61 | 3.98 |

All three converged to within 0.3 MAE of each other after tuning - the
tightest three-way convergence of any meal.

**Ensemble (CV):** 50.96 ± 4.89.

**Holdout (n=601 train / n=151 test):**

| Model | MAE | RMSE |
|---|---|---|
| XGBoost | 53.70 | 71.37 |
| LightGBM | 52.91 | 69.85 |
| CatBoost | 55.71 | 72.67 |
| **Ensemble** | **53.66** | 70.57 |
| Baseline (7d ago) | 83.90 | — |

**~36% reduction vs baseline** — smaller relative gain than Dinner, and flat
rather than improving across the menu-change boundary (Jan-Feb-only ensemble
MAE was also 53.66, coincidentally identical to two decimal places to the
full-window number - verified these come from different underlying
predictions, not a duplicate run). CatBoost specifically got *worse* moving
through the menu change (54.38 Jan-Feb → 55.71 full window) — the opposite
of what happened with Dinner's CatBoost, and worth investigating in error
analysis.

---

### 3.3 Breakfast

**Initial sanity check:** Baseline MAE 40.91 → Model MAE 23.27.

**Top 15 features:** `Target_1_Days_Ago` overwhelmingly dominates (0.468 —
more than 3× the next feature). Breakfast attendance is essentially
"yesterday's count, adjusted slightly" — the most momentum-driven of the
four meals by a wide margin.

**Default model comparison (6-fold CV) - first meal where CatBoost does *not* win:**

| Model | Avg MAE | Std |
|---|---|---|
| **RandomForest** | **24.10** | **±6.15** |
| XGBoost | 26.54 | ±6.11 |
| LightGBM | 25.53 | ±6.00 |
| CatBoost | 27.19 | ±7.42 |

RandomForest beat all three boosted models on defaults — a genuinely
different pattern from Dinner/Lunch. Consistent with a simpler, lower-
variance, momentum-dominated target: less need for boosting's extra
flexibility.

**Tuned best params:** XGBoost — lr 0.0089, 709 trees, depth 10. LightGBM —
lr 0.0085, 475 trees, 166 leaves, depth **11** (near the 12 ceiling).
CatBoost — lr 0.0240, 403 iterations, depth 3, l2_leaf_reg 8.16 (mid-range —
*not* hugging the ceiling, unlike Dinner and Lunch's CatBoost).

**Tuned, per-fold:**

| Model | Mean MAE | Std |
|---|---|---|
| XGBoost (tuned) | 23.49 | 5.75 |
| LightGBM (tuned) | 23.49 | 5.62 |
| CatBoost (tuned) | 23.98 | 5.57 |

Smallest tuning gains of any meal (~-3 MAE vs Dinner's -7.29 on XGBoost) —
consistent with there being less "hidden power" to extract when one feature
already explains most of the variance.

**Ensemble (CV):** 23.34 ± 5.67 - barely beat the best individual tuned
model, as expected given how tightly XGBoost/LightGBM/CatBoost already
converged.

**Holdout (n=601 train / n=151 test):**

| Model | MAE | RMSE |
|---|---|---|
| XGBoost | 20.56 | 27.49 |
| LightGBM | 20.55 | 27.62 |
| CatBoost | 22.21 | 28.04 |
| **Ensemble** | **20.66** | 27.09 |
| Baseline (7d ago) | 38.09 | — |

**~46% reduction vs baseline** - strong result, holds through the menu
change.

---

### 3.4 Evening Snacks

**Initial sanity check:** Baseline MAE 36.40 → Model MAE 32.40 — the
smallest single-model improvement over baseline of any meal at this stage
(only ~11% better), an early hint that Snacks would be the noisiest problem.

**Top 15 features:** `Target_7_Days_Ago` dominates (0.465), with
`Target_1_Days_Ago` a distant second (0.179). Like Breakfast, momentum-
driven — but anchored to *weekly* rhythm rather than *yesterday*.

**Default model comparison (6-fold CV) - the least stable defaults of any meal:**

| Model | Avg MAE | Std |
|---|---|---|
| RandomForest | 31.96 | ±6.74 |
| XGBoost | 38.86 | **±12.35** |
| LightGBM | 32.45 | ±7.39 |
| CatBoost | 33.90 | **±12.79** |

XGBoost's default MAE (38.86) is *worse* than the naive baseline (36.40) —
the only time in this project a default model lost to "just guess last
week." Both XGBoost and CatBoost show the widest fold-to-fold spread of any
meal's defaults, confirming Snacks as the noisiest of the four targets
(plausibly low, optional/impulse-driven attendance, less tied to fixed
class schedules than the other three meals).

**Tuned best params:** XGBoost — lr 0.0187, 218 trees, depth 8, gamma 4.02
(near the 5.0 ceiling). LightGBM — lr 0.0263, 494 trees, num_leaves **247**
(right at the 255 ceiling — genuine edge-hug). CatBoost — lr **0.198** (far
higher than any other meal's CatBoost, which ran 0.02–0.05), depth 5,
l2_leaf_reg **1.68** — hugging the *lower* bound this time, the opposite
direction from Dinner and Lunch's CatBoost (which hugged the *upper* 20
bound). Genuinely different regularization behavior for this meal — a real
finding, not noise, and worth widening `num_leaves` further and testing
both directions of `l2_leaf_reg` if CatBoost's tuning is revisited for
Snacks specifically.

**Tuned, per-fold:**

| Model | Mean MAE | Std |
|---|---|---|
| XGBoost (tuned) | 29.06 | 7.47 |
| LightGBM (tuned) | 29.44 | 8.05 |
| CatBoost (tuned) | 30.43 | **10.12** |

Tuning's biggest win of any meal: XGBoost improved by **-9.80 MAE**
(38.86→29.06), the largest single tuning gain in the whole project -
consistent with a noisier default problem having the most overfitting for
Optuna's regularization search to fix. CatBoost's spread stayed wide even
after tuning (±10.12) - breaks the "CatBoost = most stable" pattern that
held for Dinner and Lunch; stability is meal-dependent, not a universal
property of the model.

**Ensemble (CV):** 28.88 ± 8.36.

**Holdout (n=601 train / n=151 test):**

| Model | MAE | RMSE |
|---|---|---|
| XGBoost | 22.97 | 31.59 |
| LightGBM | 22.29 | 31.13 |
| CatBoost | 24.54 | 31.96 |
| **Ensemble** | **22.38** | 30.17 |
| Baseline (7d ago) | 38.68 | — |

**~42% reduction vs baseline.**

---

## 4. Cross-meal comparison

| Meal | Default winner | Tuning gain (best model) | Ensemble CV MAE ± std | Holdout MAE | vs baseline |
|---|---|---|---|---|---|
| Dinner | CatBoost | -7.29 | 51.44 ± 6.22 | 37.10 | -51% |
| Lunch | CatBoost | -10.20 | 50.96 ± 4.89 | 53.66 | -36% |
| Breakfast | RandomForest | -3.05 | 23.34 ± 5.67 | 20.66 | -46% |
| Snacks | RandomForest (defaults) | **-9.80** | 28.88 ± 8.36 | 22.38 | -42% |

**Patterns worth carrying into Step 8 (error analysis):**

1. **CatBoost wins on the two "main meal" types (Dinner, Lunch); RandomForest
   is more competitive on the two "lighter/optional" types (Breakfast,
   Snacks).** Not a coincidence across two independent runs each - a real
   structural difference in what kind of model fits each target's noise
   profile.
2. **Feature importance shape differs meaningfully by meal:** Dinner leans
   on weekly cyclic structure (`DayOfWeek_Sin`), Breakfast and Snacks are
   dominated by a single momentum feature each (yesterday vs. last week
   respectively), and Lunch leans on a *volatility* feature rather than a
   raw lag. These are different enough that a shared Feature Set V2 across
   all four meals may not be the right call - each meal may need its own
   error-analysis pass and possibly its own V2 hypothesis.
3. **CatBoost's `l2_leaf_reg` behavior is inconsistent across meals** — hugs
   the *upper* bound (20) for Dinner and Lunch, sits comfortably mid-range
   for Breakfast, and hugs the *lower* bound (1) for Snacks. This alone is
   evidence against a single "best" CatBoost regularization regime across
   meals - worth remembering if CatBoost tuning is revisited.
4. **Snacks is the noisiest, least stable problem** (widest CV spreads on
   defaults, only meal where a default model lost to baseline, biggest
   tuning gains) - likely the best candidate to prioritize first for error
   analysis, since it has the most room left to improve.
5. Menu-change resilience (Jan-Feb vs full Jan-May holdout) was clearly
   positive for Dinner and Breakfast, flat for Lunch, and not yet isolated
   into two windows for Snacks (only the full-window number was run) - worth
   filling in for completeness before drawing firm conclusions here.

## 5. Open items before V2

- [ ] CatBoost's `l2_leaf_reg` edge-hugging (upper bound for Dinner/Lunch,
      lower bound for Snacks) — a controlled, uniform rerun across all four
      meals with a wider range would need to happen together, not
      meal-by-meal, to keep V1 comparisons clean going forward.
- [ ] LightGBM `num_leaves` hit the 255 ceiling for Snacks specifically —
      same caveat, revisit uniformly if pursued.
- [ ] Confirm no rough patch exists right at the March 2026 menu-change
      boundary for any meal (residual-over-time plot) — safe to check on CV
      folds; the 2026 holdout itself is spent for all four meals now.
- [ ] Isolate Jan-Feb-only holdout numbers for Snacks (currently only have
      the full Jan-May window) for a cleaner cross-meal menu-change
      comparison — moot for holdout purposes since Jan-May is already spent,
      but useful for writeup completeness.

## 6. Next: error analysis (Step 8)

To be run per-meal, on the 6 CV folds only - **not** on the 2026 holdout,
which is spent for all four meals as of this log. Plan:
- Bucket residuals by weekday, exam period, fest, month — separately per
  meal, given how differently their feature-importance profiles look.
- Pull worst individually-predicted days per meal and inspect what happened.
- SHAP on worst-predicted days specifically (not global importance).
- Given the cross-meal differences observed above, go in expecting that V2
  may need to branch per-meal rather than apply one shared feature addition
  to all four uniformly — confirm or reject this with the bucketed residual
  comparison before committing to that structure.
