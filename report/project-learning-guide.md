# A beginner's guide to this project

This is a study document, not part of the dissertation. It exists so you can genuinely
understand every piece of this project from the ground up, in the order it was actually built,
and test yourself as you go. Read it top to bottom the first time. Come back to individual
sections before the viva.

Every technical term gets explained in plain language, with an analogy, before it gets used.
Nothing is assumed.

---

## 1. The problem and the data

### What GDP is, and what "growth" means

GDP (Gross Domestic Product) is a single number that tries to add up the total value of everything
produced in a country over a period of time. Think of it as the country's total "output" for that
period, in money terms.

GDP growth is how much bigger (or smaller) that number got compared to the previous period. If GDP
was 100 last quarter and 101 this quarter, growth is 1%. If it was 100 and then 98, growth is
negative 2%, the economy shrank.

This project uses UK GDP growth measured quarter on quarter (QoQ): each number is "how much did
the economy grow compared to the immediately previous three-month period", not compared to a year
ago. It is also:

- **Real**, meaning adjusted for inflation. If prices went up 3% and the raw money value of output
  went up 3%, real growth is 0%, nothing actually got produced, prices just rose. Real growth
  strips that out. The technical name for this in the ONS data is CVM, chained volume measures.
- **Seasonally adjusted (SA)**, meaning the normal within-year pattern (shops always sell more in
  December, farms always harvest more in autumn) has been smoothed out, so you are comparing
  genuine change, not just "it's Christmas again".

This exact combination, real, quarter on quarter, seasonally adjusted, is what the Office for
National Statistics (ONS) publishes under the series code IHYQ. That is the target this whole
project tries to forecast.

### What "forecasting" means here

Forecasting means: using information available up to and including this quarter, predict next
quarter's GDP growth, before it happens. This is called **one-step-ahead** forecasting: you are
only ever trying to see one step into the future, not five years out. Section 2 explains exactly
how "one step ahead" is enforced in the code, because it turns out to be easy to get subtly wrong.

### What the 17 features are and why each was chosen

A "feature" is just a piece of information the model is allowed to look at when making its
prediction. This project uses 17 columns in total: the target itself (GDP growth, which the model
also gets to see its own past values of), 10 other raw predictors, and 6 engineered features built
from the raw ones.

The 10 raw predictors, and why each earns its place:

| Feature | What it is | Why it might matter for forecasting GDP |
|---|---|---|
| unemployment_rate | Percentage of people who want work but do not have it | A classic leading indicator: unemployment usually rises before or during a slowdown |
| cpi_inflation | Consumer Price Index inflation, how fast prices are rising | High or unstable inflation often coincides with weaker growth |
| trade_balance | Exports minus imports | Trade is a direct component of GDP itself |
| gfcf_growth | Gross Fixed Capital Formation growth, how much businesses are investing in buildings, machinery, etc. | Investment today often shows up as growth tomorrow |
| govt_consumption_growth | Growth in government spending | Government spending is also a direct component of GDP |
| bank_rate | The Bank of England's interest rate | Higher rates cool borrowing and spending, a policy lever that affects growth with a lag |
| gbp_usd_rate | The pound's exchange rate against the US dollar | Affects how competitive UK exports are and how expensive imports are |
| brent_oil | The price of Brent crude oil | The UK imports and uses a lot of energy; oil price shocks ripple through the whole economy |
| business_confidence | A survey-based index of how optimistic businesses are | Confidence often moves before actual investment or hiring does |
| consumer_confidence | A survey-based index of how optimistic households are | Confidence often moves before actual spending does |

The 6 engineered features (built from the raw data, explained fully in Section 2):
gdp_lag_1, gdp_lag_4, gdp_rolling_mean_4q, gdp_yoy, business_confidence_rolling_mean_4q, and
yield_curve_slope.

### What "quarterly" and "104 observations" mean

A quarter is three months: Q1 is January to March, Q2 is April to June, Q3 is July to September,
Q4 is October to December. GDP is published quarterly, not monthly or daily, because it takes time
to add up an entire economy's output.

This project covers 2000 Q1 through 2025 Q4. Count the quarters: 26 years times 4 quarters is 104.
That is the entire dataset: **104 rows, one per quarter, over 26 years**. This is a genuinely small
dataset by machine learning standards (many ML problems have millions of rows), and that smallness
shapes almost every methodological decision later in the project, including why particular models
were chosen and why cross-validation had to be handled carefully.

### What a "regime" is

A regime is a stretch of time where the economy was behaving in a broadly similar way, as opposed
to the stretches before and after it. The idea is that "the economy in 2005" and "the economy in
2020" are not really the same kind of environment, one was calm, one was a global health
emergency, so a model's behaviour might reasonably differ between them.

This project splits the 104 quarters into six regimes:

| Regime | Period | Quarters |
|---|---|---|
| Pre-GFC Stability | 2000 Q1 to 2008 Q1 | 33 |
| Global Financial Crisis (GFC) | 2008 Q2 to 2009 Q3 | 6 |
| Post-GFC Recovery | 2009 Q4 to 2016 Q2 | 27 |
| Brexit Transition | 2016 Q3 to 2019 Q4 | 14 |
| COVID-19 Shock | 2020 Q1 to 2021 Q2 | 6 |
| Post-COVID Recovery | 2021 Q3 to 2025 Q4 | 18 |

Notice two of these (GFC and COVID) only have 6 quarters. That is a genuinely small sample even by
this project's already-small standards, and it comes up again and again, in evaluation, and
especially in the SHAP stability analysis in Section 8.

### What a "structural break" is

A structural break is a point in time where the underlying statistical relationship in the data
changes, not just a single unusual data point, but a shift in the rules of the game. Imagine you
are tracking how many ice creams a shop sells against the temperature outside. If the shop then
puts up a big price rise, the *relationship* between temperature and sales might shift, that price
rise is a structural break in that relationship.

The six regime boundaries above were not picked by eye. They were justified two ways: from
economic literature and history (everyone agrees 2008 was a financial crisis), and statistically,
using formal break-detection tests (the Chow test and the Bai-Perron method, both explained
briefly here since they are not the main modelling event of the project): these are statistical
tests that look at a time series and ask "does the data look like it was generated by one
consistent process the whole way through, or does it look like the process changed at some point,
and if so, roughly where?" A third test, ICSS, does the same thing but specifically for volatility
(how much the series jumps around) rather than for the average level.

### Check your understanding: Section 1

**Q1: If GDP growth is reported as 0.6% for a quarter, and inflation that quarter was also
running high, does the 0.6% already account for the inflation?**
A: Yes. The target series used in this project is real (CVM), meaning it is already adjusted for
inflation. 0.6% growth means the economy produced 0.6% more *stuff*, not just 0.6% more money
that is partly explained by higher prices.

**Q2: Why does the project use quarterly data with only 104 rows instead of, say, monthly data
with over 300 rows?**
A: Because GDP itself is only published quarterly by the ONS. You cannot manufacture a genuine
monthly GDP figure that does not exist; some of the predictor features are monthly and get
averaged up to quarterly to match. The small row count is a real constraint the whole project has
to work around, not a choice that could easily be avoided.

**Q3: Why does it matter that GFC and COVID-19 Shock only have 6 quarters each?**
A: Any statistic (an average, a correlation, a model's error rate) computed from only 6 data
points is much less reliable than one computed from 33. The project has to flag results from
these two regimes as uncertain and back them up with extra checks (bootstrap confidence
intervals, covered in Section 5), rather than reporting them as if they were as solid as the
larger regimes.

---

## 2. Features and preprocessing

### Lags: using the past to help predict the future

A "lag" feature is simply: the value of something, but from an earlier point in time, placed
alongside today's row. If GDP growth was 0.1% last quarter, then "GDP growth lagged one quarter"
for this quarter's row is 0.1%.

This project uses two GDP lags:
- **gdp_lag_1**: GDP growth from one quarter ago.
- **gdp_lag_4**: GDP growth from four quarters (one year) ago, useful because economies often have
  a yearly rhythm.

Analogy: if you were trying to guess tomorrow's weather, "what was the temperature yesterday" and
"what was the temperature on this exact date last year" are both lag features. Neither is a
guess about tomorrow directly, they are past facts you are allowed to use to help make that guess.

### Rolling means: smoothing out the noise

A rolling mean (also called a moving average) is the average of the last N values, recalculated
at every point as time moves forward. This project uses:
- **gdp_rolling_mean_4q**: the average GDP growth over the last 4 quarters.
- **business_confidence_rolling_mean_4q**: the average business confidence over the last 4
  quarters.

Analogy: instead of judging a student's ability from just their last exam score (which could be a
fluke, good or bad), you look at their average over the last four exams. A rolling mean does the
same thing for a data series, smoothing out one-off blips so the underlying trend shows through.

The technical detail that matters here: this project's rolling mean is "right-closed", meaning at
any given quarter, it only ever looks backward (the current quarter and the three before it), never
forward. That is essential, and connects directly to the "leakage" idea covered below.

### The year-on-year (YoY) feature

**gdp_yoy** compounds GDP growth over the trailing four quarters into a single "how much has the
economy grown over the last full year" figure. It answers a slightly different question than the
rolling mean of growth rates: it is the actual compounded annual change, not just an average of
quarterly rates.

### Yield curve slope

This one needs two building blocks first.

A **gilt** is a UK government bond, essentially an IOU: you lend the government money, and it pays
you back later with interest. The **yield** on a gilt is the effective interest rate it pays.
Gilts come in different maturities, how long you have to wait to get your money back: a 2-year
gilt pays back in 2 years, a 10-year gilt in 10.

Normally, lending money for longer requires a higher interest rate to compensate you for the
extra risk and waiting time, so the 10-year yield is usually higher than the 2-year yield. The
**yield curve slope** used in this project is exactly that: the 10-year yield minus the 2-year
yield.

Why it matters for GDP forecasting: when that slope goes negative (short-term yields higher than
long-term ones, called an "inverted yield curve"), it has historically been one of the more
reliable warning signs that a recession is coming, because it suggests investors expect central
banks to cut rates in future in response to a weakening economy.

### The one-step-ahead shift, and why it is the single most important design choice in the project

This is the idea that everything else depends on, so it is worth being completely precise about
it.

Every row of the feature table represents one quarter, call it quarter t. That row contains: the
raw predictors observed during quarter t, and the engineered features (lags, rolling means)
computed using data up to and including quarter t.

The target the model is trained to predict from that row is **not** GDP growth in quarter t
itself. It is GDP growth in quarter **t+1**, the very next quarter. In code this is written as
`y = gdp_growth.shift(-1)`, "shift the growth column back by one so that each row's features line
up with next quarter's growth."

Analogy: imagine a very structured weather forecaster who is only ever allowed to write down
today's temperature, humidity, and wind speed, and from those, guess tomorrow's temperature. They
are never shown tomorrow's actual conditions while making the guess, that would be cheating, and
also pointless, since the entire point is to make a genuine prediction about something you do not
yet know.

This detail became critically important later in the project (Section 9 tells that story in
full): it means a row built from quarter t's information forecasts quarter t+1, not quarter t
itself. Get that backwards, and you end up forecasting the wrong quarter entirely while thinking
you forecast the right one.

### What "leakage" means and why it must be prevented

**Leakage** is when information that would not actually be available at prediction time sneaks
into the training process, making a model look far better than it really is. It is called leakage
because it is like a hidden crack letting future information leak backward into the past.

Two concrete places this project deliberately guards against it:

1. **Scalers fit per fold.** Ridge regression needs its input features scaled (explained in
   Section 3). If you scale using statistics (mean, standard deviation) computed from the *entire*
   dataset, including data that is supposed to be in the future relative to a given test point,
   you have let a tiny bit of future information leak into how the past gets processed. The fix:
   refit the scaler from scratch inside every single cross-validation fold, using only that fold's
   training data.
2. **The rolling mean and lag windows never look forward.** As mentioned above, they are
   "right-closed": strictly today and earlier, never tomorrow.

Analogy: leakage is like a student who is meant to be tested on material up to Chapter 5, but
their revision notes accidentally include a worked answer from Chapter 6. They will do brilliantly
on the practice test and then struggle in the real exam, because the practice test was never a
fair measure of what they actually know.

### What "frozen dataset" and "vintage" mean

The **frozen dataset** is `data/processed/final_dataset.parquet`, the exact 104-row table every
model in this project was trained and evaluated on. It was built once and then locked: no new rows
get added to it, ever, even after new quarters of real data become available.

**Vintage** refers to the fact that government statistics get revised after they are first
published. The ONS publishes a first estimate of GDP growth about six weeks after a quarter ends,
then revises that figure (sometimes more than once) as better data comes in. "2025 Q1 growth" is
not one fixed number for all time, it depends on which vintage (which release date) you ask.

This stopped being a hypothetical concern partway through the project: when the 2026 predictor
data was downloaded fresh (Section 9), it turned out ONS had revised three quarters that were
already sitting in the frozen dataset (2024 Q1 moved from 0.8% to 0.7%, 2024 Q4 from 0.3% to 0.4%,
2025 Q1 from 0.7% to 0.6%). The frozen dataset still shows the original numbers, on purpose,
because the whole point of freezing it is that every model was trained and evaluated on one
specific, unchanging vintage of the data. If you let the numbers underneath shift after the fact,
you can never cleanly compare one experiment to another again.

### Check your understanding: Section 2

**Q1: A model's feature row for quarter 2019 Q3 predicts GDP growth for which quarter?**
A: 2019 Q4, the very next quarter. The one-step-ahead shift means quarter t's features predict
quarter t+1's growth, never quarter t's own growth.

**Q2: Why is it wrong to compute a scaler's mean and standard deviation once, using the whole
104-quarter dataset, and then apply it inside every cross-validation fold?**
A: Because that single global scaler would have been calculated partly using data from each
fold's future test quarters. That is leakage, it lets a small amount of future information shape
how the past gets processed, making the evaluation overly optimistic.

**Q3: Why does the frozen dataset keep the "wrong" (now-revised) GDP figures for 2024 Q1, 2024 Q4,
and 2025 Q1, instead of updating them?**
A: Because "frozen" means every experiment in the project was run against one fixed, unchanging
version of the data. Updating it after the fact would mean past results are no longer comparable
to new ones, and it would also be a subtle form of hindsight: at the time those models were
trained, the revised figures did not exist yet.

---

## 3. The four models, explained from scratch

Four different modelling approaches were compared. Each represents a genuinely different idea
about how to turn the features into a prediction.

### ARIMA

ARIMA stands for AutoRegressive Integrated Moving Average. Unpack that piece by piece.

**Autoregression** means predicting a series using its own past values, nothing else. "GDP growth
next quarter is some weighted combination of GDP growth in recent past quarters" is a purely
autoregressive idea. ARIMA in this project is univariate, it only ever looks at the GDP growth
series itself, none of the 16 other features.

Analogy: predicting your weight next month purely from your last few months' weights, without
looking at diet, exercise, or anything else.

**Stationarity** is a property a time series either has or does not have: a series is stationary
if its statistical properties (average level, how much it varies) stay roughly constant over
time, rather than drifting or trending. Many classic statistical methods, ARIMA included, work
much better, and their assumptions only really hold, on stationary series.

**The ADF test** (Augmented Dickey-Fuller test) is a formal statistical test that checks whether a
series is stationary. It produces a test statistic and a p-value (p-values are explained properly
in Section 5). For this project's GDP growth series, the ADF test returned a statistic of -9.815
with p < 0.001, strong evidence that the series is already stationary. That matters because it
means the series does not need "differencing" (see below) before ARIMA is applied.

**The (p, d, q) order** describes exactly how an ARIMA model is built:
- **p** is how many past values of the series it regresses on (the autoregressive part).
- **d** is how many times the series needs to be "differenced" (replaced with the change from one
  point to the next) before it becomes stationary. Since the ADF test already found the series
  stationary, d = 0 here, no differencing needed.
- **q** is how many past forecast errors it also incorporates (the "moving average" part).

Because the usual automatic tool for picking (p, d, q), a package called `pmdarima`, was not
available on the Python version used for this project, the order was instead chosen by a manual
grid search: trying a range of (p, d, q) combinations and picking whichever minimises AIC (Akaike
Information Criterion, a standard score that rewards a good fit but penalises unnecessary
complexity, so it does not just always pick the most complicated option).

### Ridge regression

Ordinary linear regression tries to draw the best straight-line (or flat-plane, with many
features) relationship between the features and the target. With only 103 usable training rows
and 17 features, plain linear regression can overfit badly, latching onto coincidental patterns
that will not hold up on new data.

**Regularisation** is a technique that discourages a model from assigning huge importance to any
one feature, by adding a penalty for large coefficients. **Ridge regression** is linear regression
with this specific kind of penalty (technically, an L2 penalty on the size of the coefficients)
added in.

Analogy: imagine grading an essay where using any one single word excessively is penalised, even
if that word is genuinely useful. It nudges the writer toward a more balanced use of many words
rather than leaning entirely on one, which tends to generalise better to new topics.

Ridge needs its input features **scaled** first (put on a comparable numeric range, like
converting both centimetres and kilometres into metres before comparing them) because the penalty
is applied to the size of coefficients, and a feature measured in tiny units (like a percentage
that moves by 0.1) would otherwise need an artificially huge coefficient just to matter as much
as a feature measured in large units (like an oil price in the hundreds), unfairly distorting how
the penalty falls.

### XGBoost and LightGBM

Both of these are **gradient boosting** methods built from **decision trees**.

A decision tree is a flowchart of yes/no questions that ends in a prediction. "Is unemployment
above 5%? If yes, is bank rate above 3%? If yes, predict 0.1% growth." A single tree, on its own,
is usually a weak, rough predictor.

**Gradient boosting** builds many small trees one after another, where each new tree is trained
specifically to correct the mistakes the trees built so far are still making. Combined together,
often dozens or hundreds of small trees end up making a much stronger prediction than any single
one could.

Analogy: instead of one person marking an exam and giving a final grade, imagine a sequence of
markers, where each new marker's whole job is to spot what the previous markers got wrong and
nudge the grade toward the truth. After enough rounds, the combined grade is far more accurate
than any single marker's.

XGBoost and LightGBM are two different, competing software implementations of this same gradient
boosting idea, with different internal engineering choices. This project treats them as two
separate models to compare, since either could plausibly perform better on any given dataset.

**Why they are called "black boxes":** a single decision tree is easy for a human to read start to
finish. A gradient boosting model made of, say, 50 or 200 trees, each correcting the last, is not
something a person can read through and understand directly, there are too many interacting parts.
The overall model can still make excellent predictions, but *why* it made any particular prediction
is no longer obvious just by looking at it. This is exactly the problem SHAP (Section 7) exists to
address.

Trees do not need feature scaling. A tree just asks "is this value above or below some threshold",
so whether that threshold is in small units or large units makes no practical difference, unlike
Ridge.

### Why a three-way comparison, not four separate models in a vacuum

The four models were not compared arbitrarily. They were deliberately chosen to isolate two
separate questions:

1. **ARIMA vs Ridge**: ARIMA only sees GDP's own past. Ridge sees all 17 features. Comparing them
   isolates the value of adding the extra macroeconomic features at all, regardless of whether the
   relationship is a straight line or something more complex.
2. **Ridge vs XGBoost/LightGBM**: Ridge assumes a straight-line relationship between features and
   the target. The gradient boosting models can capture bends, thresholds, and interactions.
   Comparing them isolates the value of allowing non-linear relationships, holding the feature set
   fixed.

This structure means a simple "boosting won" or "boosting lost" result is never enough on its own,
the design lets you say specifically *why*, whether it was the extra features, the non-linearity,
both, or neither.

### Check your understanding: Section 3

**Q1: If the ADF test on GDP growth had come back with a high p-value (not significant), what
would that have meant for choosing ARIMA's d parameter?**
A: A high (not significant) p-value would suggest the series is not stationary, meaning it would
likely need differencing before ARIMA is applied, so d would probably need to be 1 or higher
instead of 0.

**Q2: Why does Ridge regression need its features scaled, but XGBoost and LightGBM do not?**
A: Ridge penalises the size of coefficients, so features on very different numeric scales would be
penalised unfairly unless everything is put on a comparable range first. Trees only ever ask
"above or below this threshold", so the actual scale of a feature does not change how a tree uses
it.

**Q3: What specific question does comparing Ridge against XGBoost and LightGBM answer, that
comparing ARIMA against Ridge does not?**
A: Ridge vs the gradient boosters isolates whether allowing non-linear, more complex relationships
between the features and GDP growth actually helps, since both see the same 17 features, they just
differ in how flexible the relationship they can learn is.

---

## 4. Cross-validation

### What cross-validation is, and why it is needed

If you train a model and then check how well it predicts the exact same data it was trained on,
you learn almost nothing useful, of course it does reasonably well, it has already seen the
answers. **Cross-validation (CV)** is a family of techniques for checking how a model performs on
data it did not see during training, to get an honest estimate of how it will do on genuinely new,
future data.

Analogy: it is the difference between grading a student on questions identical to their homework
answers, versus grading them on a fresh exam covering the same topics. Only the second one tells
you what they actually learned.

### What a "fold" is

A **fold** is one round of a train-then-test split: some rows are used to train the model, a
different set of rows (never seen during that round's training) are used to test it. Doing this
several times, with different splits each time, gives a more reliable overall picture than relying
on a single split.

### Expanding-window CV, the primary scheme

Because this is time series data, folds cannot be chosen randomly, that would let a model "see the
future" relative to some of its test points, exactly the kind of leakage described in Section 2.

**Expanding-window CV** instead always trains on everything up to a point in time, and tests on
the few quarters immediately after. Each successive fold pushes that point forward and grows the
training window (hence "expanding"). Concretely in this project: the first fold might train on
quarters 1 to roughly 72 and test on the next 4, the next fold trains on 1 to 76 and tests on the
next 4, and so on, 8 folds in total, always walking forward through real time and never peeking
ahead.

Analogy: this mimics genuinely being a forecaster living through history: today you only have data
up to today, tomorrow you will have one more day of data than you do right now, and so on. It is
the most realistic simulation of "how would this model actually have performed if you had been
using it in real time".

### Regime-aligned CV, the secondary scheme

Expanding-window CV has a real weakness for this project: because its early folds train on such a
large chunk of history, its test folds only ever fall in the later quarters. It never actually
tests the model on quarters from the Global Financial Crisis or Post-GFC Recovery regimes at all,
since those are all comfortably inside the training portion by the time any fold gets to testing.

**Regime-aligned CV** is built specifically to guarantee every regime gets tested at some point:
each fold trains on the regimes seen so far and tests on the regime that comes next
chronologically, walking through all six regimes in order.

### Why two schemes, not one

They answer genuinely different questions. Expanding-window answers "how would this model have
performed walking forward through history, quarter by quarter, the way a real forecaster would
experience it". Regime-aligned answers "how does this model cope with a kind of economic
environment it has not seen examples of yet, including the Global Financial Crisis and Post-GFC
Recovery, which expanding-window's test folds never touch". Neither one alone tells the full
story, which is why both are reported side by side rather than picking just one.

One subtlety worth knowing for the viva: under regime-aligned CV, the "n" (sample size) reported
for a regime in a results table counts *prediction instances*, not unique quarters, because a
regime can get tested more than once across different folds. A regime with only 6 unique quarters
can show n=24 in a results table, that is not a mistake, it reflects how many times, across all
folds, a prediction was made that fell in that regime.

### Check your understanding: Section 4

**Q1: Why can't cross-validation folds for this project be chosen by randomly shuffling all 104
quarters and splitting them?**
A: Random shuffling would let some training rows come from *after* some test rows in time, meaning
the model could effectively be trained partly on the future relative to what it is being tested
on. That is leakage, and it makes the evaluation dishonestly optimistic.

**Q2: Why does expanding-window CV never produce a test fold that falls in the Global Financial
Crisis regime?**
A: Its very first fold already trains on enough history to satisfy the minimum training size
requirement, and by that point the GFC and Post-GFC Recovery regimes are already comfortably
inside the training portion, so no test fold ever lands there.

**Q3: If a table shows a regime with 6 unique quarters but n=24 under regime-aligned CV, is that
an error?**
A: No. Under regime-aligned CV, a regime can be tested across multiple folds, so n counts how many
times a prediction was made for a quarter in that regime, not how many distinct quarters exist. The
effective sample size for judging reliability is still the unique-quarter count (6), not the
reported n.

---

## 5. Evaluation

### The four accuracy metrics, in plain terms

Each of these takes a set of predictions and a set of actual values, and boils the difference down
into a single number.

- **RMSE (Root Mean Squared Error)**: take each prediction's error (predicted minus actual), square
  it, average all the squared errors, then take the square root. Squaring means large errors get
  punished disproportionately harder than small ones. Lower is better. It is in the same units as
  GDP growth itself (percentage points).
- **MAE (Mean Absolute Error)**: take each error's absolute size (ignore whether it was too high or
  too low), and average them. Simpler to interpret than RMSE, and treats a big error and several
  smaller errors of the same total size more proportionately, since it does not square anything.
- **MASE (Mean Absolute Scaled Error)**: MAE, but divided by the MAE of the simplest possible naive
  guess (predicting "next quarter will be the same as this quarter"). A MASE below 1.0 means the
  model beats that naive guess; above 1.0 means it is actually worse than just guessing "no
  change". This project follows the standard Hyndman and Koehler definition and always uses the
  same naive denominator so MASE values are comparable across every table.
- **R2 (R-squared)**: how much of the variation in the actual values the model's predictions
  explain, relative to just always predicting the average value. R2 of 1.0 would mean perfect
  prediction. R2 of 0.0 means the model does no better than always guessing the average. **R2 can
  go negative**, and that specifically means the model is doing *worse* than simply guessing the
  average every time, which is a real, meaningful, if unflattering, result, not a sign of a
  calculation error.

### The Diebold-Mariano test

Once you have two models' error scores, a natural question is: is one model's better score a real,
reliable difference, or could it easily have happened by chance on this particular set of test
quarters? The **Diebold-Mariano (DM) test** is a formal statistical test built specifically to
answer that question for pairs of forecasting models.

This project uses a specific refinement of it, the **Harvey-Leybourne-Newbold (HLN) correction**,
which adjusts the test to behave better on small samples (recall, only around 100 quarters total),
and compares the result against a t-distribution reference with n-1 degrees of freedom rather than
the more usual reference, again because the sample is small.

### What "statistical significance" and "p-value" mean

A **p-value** is the probability of seeing a result at least as extreme as the one you actually
got, *if* there were truly no real difference at all. A small p-value (conventionally, below 0.05)
is usually taken as evidence that the observed difference probably is not just random chance.
**"Statistically significant"** is the label given to a result whose p-value clears that bar.

Analogy: if you flipped a coin 10 times and got 9 heads, the p-value answers "how likely is 9 or
more heads out of 10, if the coin were actually fair?" A very small answer would make you doubt the
coin is fair at all.

### Bonferroni correction

If you run many statistical tests at once, some will look "significant" purely by chance, just
like rolling enough dice eventually gives you a run of sixes, even with fair dice. The
**Bonferroni correction** guards against this by making the significance bar stricter in proportion
to how many tests you are running at once (in this project, six pairwise model comparisons at
once), so that the overall chance of a false alarm across the whole family of tests stays under
control.

This project reports both the raw (uncorrected) p-value and the Bonferroni-corrected one, so
nothing is hidden either way.

### Bootstrap confidence intervals

A **bootstrap confidence interval (CI)** is a way of estimating how uncertain a statistic is when
you have very little data to compute it from, without assuming any particular textbook formula
applies.

The method: take your small sample (say, the 6 quarters in the COVID-19 Shock regime), and
repeatedly draw a new sample of the same size *with replacement* (meaning the same original
quarter can be picked more than once, or not at all, in any given draw) from it. Recompute your
statistic (an error metric, a feature ranking, whatever you are interested in) on each of these
resampled draws, do this many times (1000 times in this project), and look at the spread of
results you get. The middle 95% of that spread is your 95% confidence interval.

Analogy: imagine you only got to interview 6 people about their opinion, and you want some sense of
how much that opinion might have varied if you had happened to interview a slightly different mix
of 6 people. Bootstrapping simulates exactly that, by repeatedly re-drawing from the people you
actually did interview.

This project reports bootstrap CIs specifically for the two small regimes, GFC and COVID-19 Shock,
since a single point estimate from only 6 observations, reported with no sense of its uncertainty,
would be misleadingly precise-looking.

### Check your understanding: Section 5

**Q1: A model gets an R2 of negative 0.7 on some test data. Is that a bug?**
A: No. Negative R2 is a legitimate, meaningful result: it means the model's predictions are worse
than simply always predicting the average value. It happens in this project's results and is
reported honestly rather than hidden.

**Q2: Two models are compared with a Diebold-Mariano test and the raw p-value is 0.03, but after
Bonferroni correction it becomes 0.18. Which one determines whether you call the difference
statistically significant, given six comparisons were run at once?**
A: The Bonferroni-corrected value, 0.18, since it accounts for the fact that six tests were run
simultaneously, and 0.18 is above the usual 0.05 threshold, so this particular difference would
not be called statistically significant once that correction is applied.

**Q3: Why does this project use bootstrap confidence intervals specifically for the GFC and
COVID-19 Shock regimes, and not, say, Pre-GFC Stability?**
A: GFC and COVID-19 Shock each have only 6 quarters, far too few to trust a single point estimate
without some measure of how much it could have varied. Pre-GFC Stability has 33 quarters, enough
that its point estimates are already comparatively more reliable on their own.

---

## 6. The results, and why a near-null finding is honest, not a failure

### What actually happened

Across both CV schemes, XGBoost came out with the lowest mean RMSE among the four models (2.397 on
expanding-window CV), narrowly ahead of LightGBM (2.497), with Ridge and ARIMA further behind.
However:

- Every model's overall R2 came out negative, meaning none of the four consistently beat "always
  guess the average" once you look across the whole test period.
- All six pairwise Diebold-Mariano comparisons, after Bonferroni correction, came back with a
  p-value of essentially 1.000, meaning there is no statistically reliable evidence that any one
  model is genuinely better than any other at this task.

### Why this is called a "near-null finding", and why reporting it honestly matters

A **null finding** in research means "we did not find the effect or difference we were looking
for". Here, that means "none of these four models can be shown to reliably beat the others, or to
reliably beat a naive guess". This is sometimes seen as a disappointing outcome, but it is a
genuinely useful and honest one: UK quarterly GDP, with roughly 100 data points and this particular
feature set, turns out to be a very hard forecasting problem, and pretending otherwise (for
instance, by only reporting the one CV fold or one metric that happened to look best) would be
scientifically dishonest.

This project's whole design was built to surface this cleanly rather than accidentally bury it:
multiple metrics, two CV schemes, a formal significance test with a correction for running six
comparisons at once. A result like "ARIMA wins" or "no model wins" was always treated as an equally
valid, reportable outcome from the start, not a failure of the analysis.

### Check your understanding: Section 6

**Q1: XGBoost had the numerically lowest RMSE. Does that alone mean XGBoost is proven to be the
better model?**
A: No. A lower RMSE on this particular test data does not establish statistical significance on
its own. The Diebold-Mariano tests found no statistically significant difference between any pair
of models after correcting for multiple comparisons, so XGBoost's edge could plausibly be down to
chance.

**Q2: Why is "none of the four models beats the naive baseline" a scientifically useful thing to
report, rather than something to hide?**
A: It is an honest, informative characterisation of how hard this specific forecasting problem
actually is, and it is exactly the kind of result the evaluation framework (multiple metrics, two
CV schemes, significance testing) was deliberately built to be able to detect and report cleanly,
rather than only ever reporting favourable-looking numbers.

---

## 7. SHAP

### The problem SHAP solves

Section 3 explained that gradient boosting models like XGBoost are "black boxes": they can predict
well, but a human cannot just read the model and see why it made a particular prediction. **SHAP
(SHapley Additive exPlanations)** is a method for opening that box a little: for any single
prediction, it tells you how much each individual feature pushed the prediction up or down, away
from some baseline "average" prediction.

### The game theory analogy: what a Shapley value actually is

Imagine a group of people worked together on a project and earned a prize, and now you want to
fairly divide credit for that prize among them, given that some people's contribution only really
mattered in combination with others. A **Shapley value** (named after the economist Lloyd Shapley,
who introduced the idea in cooperative game theory) is a mathematically fair way of dividing that
credit: for every possible sub-group (coalition) you could have formed without a given person,
check how much better the outcome got once you added them in, then average that "marginal
contribution" across every possible ordering in which people could have joined.

SHAP applies this exact idea to a model's features instead of people, and the "prize" is a single
prediction. For one specific prediction, a feature's SHAP value is, loosely, "how much did adding
this feature's specific value, on average across every possible combination of the other
features being present or absent, push the prediction away from the baseline".

### TreeSHAP

Computing a true Shapley value exactly, by trying every possible combination of features, is
usually far too computationally expensive to do directly. **TreeSHAP** is a specific fast
algorithm for computing SHAP values exactly (not approximately) for tree-based models like
XGBoost and LightGBM, by exploiting how their tree structure works internally. Because it is exact
for trees, it is also fully deterministic: run it twice on the same model and same data, and you
get identical numbers both times, no randomness involved. This project uses TreeSHAP exclusively,
partly for that determinism, and partly because it sidesteps a known instability problem that
affects sampling-based SHAP methods (KernelSHAP), which are approximate and can give slightly
different answers on different runs.

### Feature importance, from SHAP

Once you have a SHAP value for every feature, for every row, a natural summary is: take the
average of the *absolute* SHAP value for each feature across all the rows you care about. A
feature with a high average absolute SHAP value is one that, on average, pushes predictions around
by a lot, in either direction, it "matters" to the model's reasoning. Ranking features by this
average gives a feature importance ranking.

### The two-feature finding, and what it means

When this project computed SHAP for the best-performing gradient boosting model (XGBoost, fitted
on the full frozen 103-row training set), a striking pattern showed up: only **2 of the 17
features ever had a nonzero SHAP value, in any of the six regimes**: `gdp_growth` (the model's own
most recent quarter's growth, used as an input feature) and `gdp_lag_4` (growth from a year ago).
All 15 other features, every single macroeconomic predictor in the dataset, had an exact SHAP
value of zero, everywhere.

This was checked carefully before being trusted (the full story is in the decision log): it
matched XGBoost's own built-in, SHAP-independent feature importance exactly, ruling out a bug in
the SHAP code itself. A follow-up experiment then ruled out under-fitting: giving the model more
capacity (more trees, a higher learning rate, deeper trees) did make it start using more of the 15
other features, but its cross-validated accuracy got steadily *worse* as it did so, evidence that
those extra features were adding noise, not recovering missing signal. In short: the best,
properly cross-validated model for this dataset turns out to be, in substance, close to a
two-feature autoregression on GDP's own recent history, and the macroeconomic predictors genuinely
do not carry enough exploitable signal, at this sample size, to earn a place in that model.

### Check your understanding: Section 7

**Q1: In the group-project prize analogy, what does a person's Shapley value represent?**
A: Their fair share of the credit for the group's outcome, calculated by averaging how much better
the outcome got when they joined, across every possible order in which the group could have been
assembled.

**Q2: Why does this project use TreeSHAP specifically, rather than a more general SHAP method?**
A: TreeSHAP computes exact, deterministic SHAP values for tree-based models like XGBoost by
exploiting their internal structure, rather than approximating with random sampling the way
KernelSHAP does. That avoids a known source of run-to-run instability in SHAP results that comes
purely from the computation method itself, not from anything about the data.

**Q3: If a feature has an average absolute SHAP value of exactly zero across every regime, what
does that tell you about the model, as distinct from the real world?**
A: It tells you that this specific fitted model never uses that feature to make a difference to
its predictions, in any of the trees it built. It is a statement about how this model reasons, not
a claim that the feature has no real-world relationship to GDP at all.

---

## 8. The stability analysis

### Spearman rank correlation

Once you have a feature importance ranking for each of the six regimes (which feature is 1st most
important, 2nd, and so on, within that regime), a natural question is: how similar are these
rankings across regimes? **Spearman rank correlation** measures exactly that: it compares two
rankings and produces a single number, conventionally called rho (the Greek letter r), between -1
and 1.

- Rho = 1 means the two rankings are identical.
- Rho = -1 means one ranking is the exact reverse of the other.
- Rho near 0 means the two rankings have no meaningful relationship to each other.

Rankings, rather than the raw SHAP magnitudes, are compared deliberately: raw magnitudes shift
around simply because GDP is more volatile in some regimes than others, but rank abstracts that
away and asks only "which features matter most", not "by how much".

### The stability bands used in this project

To turn a raw rho value into a plain-language label, this project uses three bands, following
Akoglu (2018):

- **Stable**: rho > 0.6
- **Moderately stable**: 0.3 < rho ≤ 0.6
- **Unstable**: rho ≤ 0.3

### Why the result came out trivially stable

Every single one of the 15 possible regime pairs (6 regimes, so 6 choose 2 equals 15 pairs) came
back with rho = 1.000 exactly, the maximum possible value, landing every pair firmly in the
"stable" band.

At first glance this could look like the headline result of the whole novel contribution:
"explanations are perfectly consistent across every economic regime studied". But Section 7
already explained why that is the wrong way to read it: only 2 of the 17 features ever have a
nonzero SHAP value, and they are the *same* 2 features, in the *same* order, in every one of the
six regimes. Spearman correlation is comparing rankings, and if only two positions in a ranking
are ever anything other than a tie, and those two positions never swap, there is essentially
nothing left for the ranking to reorder. A perfect rho of 1.000 in that situation reflects the
model's structure (it collapsed to two features), not evidence that the model is reasoning
consistently about a rich set of features under genuinely different economic conditions.

This is exactly why the decision log entry on this finding matters, and why the project paused
this analysis pending supervisor discussion: a "perfectly stable" result that is trivial for a
structural reason is a fundamentally different finding, and needs a fundamentally different
write-up, than a "perfectly stable" result earned from a model that genuinely used a broad set of
features consistently well across every kind of economic environment.

### Check your understanding: Section 8

**Q1: A Spearman rho of 0.45 between two regimes falls into which stability band, using this
project's bands?**
A: Moderately stable (0.3 < rho ≤ 0.6).

**Q2: Why does this project compare feature *rankings* between regimes, rather than comparing the
raw SHAP values directly?**
A: Raw SHAP magnitudes naturally shift simply because GDP is more volatile in some regimes (like
COVID-19 Shock) than others, which would confound a genuine change in "which features matter" with
a change in "how big the numbers happen to be". Rankings strip away magnitude and compare only
relative importance.

**Q3: Why is a Spearman rho of 1.000 across every regime pair not automatically good news for
this project's novel contribution?**
A: Because it turned out to result from the model using only 2 of 17 features, in the same order,
everywhere, so there was essentially nothing for the ranking to reorder in the first place. A
trivially perfect stability score is a different, weaker finding than genuine evidence that a
model's rich, multi-feature reasoning holds up consistently across different economic conditions.

---

## 9. The 2026 forecast

### What was actually being tested

Separately from the main dissertation analysis, the project also produced a live, out-of-sample
test: take the best model, and see how well it actually forecasts a real quarter that had not
happened yet when the training data was frozen, then check that forecast against what genuinely
occurred.

### How the 2026 predictor data was obtained

New raw predictor data for 2026 needed downloading. One genuine bug turned up and was fixed along
the way: the Bank of England data downloader had a hardcoded date cutoff of 31 December 2025 baked
into its request, meaning it would never have pulled in 2026 data at all until that was corrected.
After fixing it, a coverage report was built to check, quarter by quarter, which of the ten raw
predictors actually had a complete quarter of real 2026 data. Q1 2026 came back fully complete (13
out of 13 predictor-quarter combinations); Q2 2026 was still mostly incomplete at the time, since
several data sources publish with a lag of a month or two.

### The subtlety that mattered most: which quarter does a "2026 Q1 forecast" actually need as input

Recall from Section 2 that a feature row built from quarter t predicts quarter t+1, never quarter
t itself. Working through what that actually means here: to forecast GDP growth **for** 2026 Q1,
the model needs a feature row built **from** 2025 Q4, the quarter immediately before it, not a row
built from 2026 Q1's own data.

This turned out to be an easy and natural mistake to almost make: the newly downloaded, complete
set of real 2026 Q1 predictor data is exactly what you would instinctively reach for to "forecast
2026 Q1". But under this project's one-step-ahead convention, that same 2026 Q1 predictor row
actually forecasts 2026 **Q2**, not Q1. The genuine 2026 Q1 forecast needed nothing newly
downloaded at all: the required feature row, built from 2025 Q4, was already sitting in the frozen
dataset the entire time.

### Handling the vintage drift correctly

Section 2 mentioned that ONS had revised three quarters already sitting in the frozen dataset by
the time this forecast was produced. Two of those revised quarters, 2024 Q4 and 2025 Q1, happened
to fall inside the exact lag and rolling-mean window needed to build the 2025 Q4 feature row
(`gdp_lag_4` reaches back to 2024 Q4; the rolling mean and year-on-year features both span 2025 Q1
through Q4). To keep the forecast's inputs consistent with what the model was actually trained on,
those lag and rolling values were read directly from the frozen dataset, not recomputed from the
freshly re-downloaded (and by then, revised) raw data.

### Producing and validating the forecast

XGBoost was refit on the full frozen 2000 to 2025 history, using the same cached hyperparameters
Sprint 3 had already selected (no re-tuning), saved as a separate model file so it stays clearly
distinct from the copies used elsewhere in the project. Predicting from the 2025 Q4 feature row
gave a forecast of **0.444%** for 2026 Q1 GDP growth.

The real ONS figure for 2026 Q1, once published, was **0.6%**. That is an error of about 0.156
percentage points. On its own, a single number like that is hard to judge, so it was read against
XGBoost's own typical error in the Post-COVID Recovery regime (the regime 2026 Q1 falls into) from
the Sprint 4 evaluation: a mean absolute error there of around 0.42 and an RMSE of around 0.51. An
error of 0.156 is comfortably smaller than both, well within the range of error this model
typically makes in this regime, rather than a surprising miss.

### Check your understanding: Section 9

**Q1: To genuinely forecast GDP growth for 2026 Q1, which quarter's feature row does the model
actually need?**
A: 2025 Q4's feature row, the quarter immediately before it, because of the one-step-ahead
convention where quarter t's features predict quarter t+1's growth.

**Q2: If you had instead fed the model the real, newly downloaded 2026 Q1 predictor data, what
would that forecast actually represent?**
A: A forecast for 2026 Q2, not 2026 Q1, since a feature row built from 2026 Q1 predicts the
quarter after it under this project's shift convention.

**Q3: Why were the lag and rolling-mean values for the 2026 Q1 forecast taken from the frozen
dataset rather than recalculated from the freshly downloaded raw data?**
A: Because ONS had revised some of the underlying GDP figures (2024 Q4 and 2025 Q1) since the
frozen dataset was built, and those exact quarters fall inside the lag and rolling-mean windows
needed for this forecast. Using the frozen values keeps the forecast consistent with the specific
data vintage the model was actually trained on.

---

## How this all fits together

Working backward through everything above: the project starts from a genuinely small, carefully
defined dataset (Section 1), builds features from it with strict discipline against
letting future information leak backward (Section 2), tries four structurally different modelling
approaches chosen specifically to isolate *why* any differences appear (Section 3), evaluates them
honestly using two complementary cross-validation schemes (Section 4) and a full toolkit of
metrics and significance testing (Section 5), and reports a genuinely near-null result without
dressing it up (Section 6). The explainability work built on top of the best model (Sections 7 and
8) then uncovered something almost as important as the original forecasting question itself: the
model everything else was built on turns out to rely on very little of the feature set it was
given, which reframes what the stability finding actually means. Section 9 showed the same
one-step-ahead discipline from Section 2 mattering all over again, in a live setting, where getting
it backwards would have quietly produced a forecast for the wrong quarter.
