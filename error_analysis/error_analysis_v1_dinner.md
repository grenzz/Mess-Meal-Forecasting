# Error Analysis V1, Dinner

## Summary

V1 gets the overall pattern right, but it has a few clear weak spots. The model
struggles the most on unusual days, meaning very high or very low attendance,
during exam periods, on Sundays, and right around vacation start and end dates.
All three models (XGBoost, LightGBM, CatBoost) fail on the same days, which
means the problem is not the model, it is the features. This document lists
everything that was checked, what was found, and what was ruled out.

This analysis was done on the 6 fold CV residuals from V1, not on the 2026
holdout.

---

## Tier 1: Foundational findings

These change how we think about the whole project.

### 1. All three models fail on the same days

Evidence:
- XGBoost, LightGBM and CatBoost all get about the same MAE (51 to 52)
- The ensemble only improves things a little bit over the best single model
- About 89% of the worst error days overlap across all three models
- How much the three models disagree with each other barely predicts how
  wrong they actually are (correlation of only 0.15)

What this means:
Three very different models are all hitting the same wall on the same days.
If it was a model problem, one of them would do noticeably better on the hard
days. Since none of them do, the problem is in the features, not the model.
More tuning or a smarter ensemble will not fix this.

### 2. The model regresses toward the mean

Evidence:
- On low attendance days, the model consistently predicts too high
  (overpredicts by 30 to 40 on average)
- On high attendance days, the model consistently predicts too low
  (underpredicts by 60+ on average)
- This shows up clearly when you plot residual against actual headcount,
  it is a strong, visible trend, not just a little noise
- The highest attendance bucket has a much bigger average error than the
  middle buckets

What this means:
The model plays it safe. It is good at predicting normal days but it does
not have enough information to know when a day is going to be unusually
busy or unusually quiet. This is a common weakness of tree based models
in general, not just specific to your data.

---

## Tier 2: Statistically strong findings

These are backed by formal statistical tests, not just looking.

### 3. There is leftover time pattern in the errors (autocorrelation)

Evidence:
- Ljung Box test on the residuals is significant at every lag checked
  (1, 3, 7, 14 days), with very small p values (around 1e-10 to 1e-20)
- Looking at bias fold by fold, it does not bounce around zero randomly,
  it sits on one side for weeks and then flips to the other side
  (for example, plus 18.7 in March to May, then minus 16.1 in September
  to November, then minus 26.3 in November to December)

What this means:
There is a slow moving pattern the model is not capturing, something like
where you are in the semester. Nothing in the current features tracks this
directly. WeekOfYear does not know where semester boundaries are, so it
does not fully capture this.

### 4. The exam period feature is too simple

Evidence:
- Days flagged as exam period have much higher error than normal days
  (MAE around 68 to 97 versus around 47 for non exam days)
- When exam period is split by the actual calendar category, end sem exams
  have almost double the error of mid sem exams
  (end sem MAE 97.5, mid sem MAE 68.0)
- This effect holds up even after controlling for attendance level, so it
  is not just because exam days happen to have low attendance

What this means:
The current single Is_Exam_Period flag is hiding a real difference between
mid sem and end sem exams. End sem is a much bigger behavior change,
probably because people are finishing up and leaving campus, not just
studying more.

---

## Tier 3: Practical engineering findings

### 5. Sunday attendance is underpredicted, even controlling for how busy the day is

Evidence:
- Checked Saturday and Sunday separately against days with similar actual
  attendance levels (not just against the whole dataset)
- Saturday's bias mostly disappeared once you compare it to other days with
  similar attendance, meaning Saturday's original bias was probably just
  because Saturdays tend to be quiet days, not a real Saturday effect
- Sunday's bias did not disappear. In every attendance bucket checked,
  Sunday is more underpredicted than other days at the same attendance
  level, and the gap does not shrink at higher attendance, if anything
  it grows

What this means:
There is a real Sunday specific effect, separate from just attendance
level. Saturday does not have this same effect, so it is not a general
weekend thing, it is specific to Sunday.

### 6. There is a lag effect right around vacation start and end

Evidence:
- Looked at the 5 days right before each vacation period starts, and the
  5 days right after each vacation period ends
- 5 days before vacation starts: model overpredicts by about 60 on average,
  error is about 40% worse than normal days
- 5 days after vacation ends: model underpredicts by about 57 on average,
  error is about 33% worse than normal days

What this means:
The model's lag and rolling average features are still anchored to the
"before" pattern when a transition happens. Right before a vacation,
attendance is probably already starting to drop, but the model does not
know that yet since its lag features are still seeing normal numbers.
Same thing in reverse right after vacation ends.

Caveat: sample size here is small. Only 2 vacation blocks started and
1 block ended inside the validation year, so this is 10 days and 5 days
respectively. The direction and general size of the effect is probably
real given how big it is, but do not treat the exact numbers as precise.

---

## Tier 4: Rejected hypotheses

These were checked and did not hold up. Writing them down so nobody wastes
time checking them again later.

### Day before a holiday
Checked MAE for the day right before a holiday (excluding the holiday
itself) against all other days. MAE was 50.0 versus 61.7 for other days,
meaning it was actually predicted slightly better, not worse. No effect.

### Day after a holiday
Checked MAE for the day right after a holiday. MAE was 60.6 versus 59.6
for other days. Basically no difference. No effect.

### Saturday bias
Originally looked like Saturday had a real bias (overpredicting by about
26 on average). Once checked against other days with similar attendance
levels, this mostly disappeared. It was mostly explained by Saturdays
tending to be low attendance days, not a real Saturday specific effect.

### July 15, 2025 spike
This was the single worst prediction of the whole year (actual 459,
predicted 204). Checked the calendar and it falls in the middle of summer
vacation, no exam, no fest, no holiday, nothing unusual flagged. Also
checked the days around it and there is no buildup or pattern nearby, it
is a single isolated spike. Treating this as random noise, not a pattern
to build a feature around. If you happen to remember something specific
that happened that day (an event, a function, anything) it might be worth
asking around, but not worth spending engineering effort chasing a single
day.

---

## V1 conclusions
 
Current tree boosting models (XGBoost, LightGBM, CatBoost) have gotten
about as much as they can out of Feature Set V1. All three land on almost
the same MAE and fail on the same days, so the limit is not model capacity,
it is the information available in the features. Further tuning or trying
new boosting models is unlikely to help much on its own. The main way
forward is adding features that explain what the current ones miss,
especially around exam type, semester timing, and unusual (very high or
very low) attendance days. It is also possible that the extreme attendance
problem needs a change in how the model is trained, not just new columns,
see the caveat on that below.
 
## V2 design priorities
 
Each row below is a candidate direction for V2, ranked by priority. Every
candidate is tied back to a specific finding above, if a feature idea does
not trace back to one of these findings, it should not be built without
new evidence first.
 
| Priority | Candidate | Parent finding | Confidence | Available now? |
|---|---|---|---|---|
| High | Split exam period into mid sem and end sem | Finding 4 | High | Yes, already have Category column |
| High | Semester position feature (something like days since semester start) | Finding 3 | High | Yes, have semester start/end dates in calendar |
| High | Something to explain extreme attendance days | Finding 2 | High (that the problem exists), Low (on what the fix is) | Partly, see caveat below |
| Medium | Vacation transition (days before/after boundary) | Finding 6 | Medium, large effect but small sample | Yes, have vacation dates |
| Medium | Sunday specific behavior | Finding 5 | Medium, confirmed but reason unknown | Yes, already have day of week |
| Low | Try new boosting models or more Optuna trials | Finding 1 | High that this will not help | Not worth doing, already saturated |
 
On the extreme attendance row: the finding that low attendance days
are overpredicted and high attendance days are underpredicted (Finding 2)
is the biggest failure mode found, but the fix might not be just a new
feature. It could need a change in how the model is trained, like quantile
regression or giving more training weight to rare high and low attendance
days. Feature engineering can still help some (a flag for "this looks like
an unusual week" might reduce it a little).

---
