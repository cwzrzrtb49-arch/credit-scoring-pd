# PD Model in R

Academic project for the application of supervised learning techniques to the problem of credit risk.
Built on the German Credit dataset (UCI Statlog, 1,000 loans, 20 predictors).

## Objective
Calculate the probability of default (PD) of the borrower, assess the model using
standard credit-risk metrics and interpret the result in terms of the Basel framework
(Expected Loss = PD x LGD x EAD).

## Approach
1. Data preprocessing: target definition (1 = default), aggregation of rare categories
2. Stratified train/test split 70%/30%
3. Logistic regression (the standard method for PD or scorecard models)
4. Lasso-penalized logistic regression (cross-validated) for feature selection
5. Model assessment: AUC, Gini, KS, confusion matrix at different cut-offs
6. Calibration: predicted PD vs observed default rate by ratings class
7. Expected Loss on the test portfolio
8. Monte Carlo simulation (10,000 iterations) of the portfolio loss distribution:
mean loss, 99% VaR and 99% Expected Shortfall

## Results
Test dataset: 300 loans (stratified 70%/30% split). Plots and tables are available in the `output` directory.

|Metric|Logistic|Lasso|
|---|---|---|
|AUC|0.752|0.746|
|Gini|0.504|0.493|
|KS|-|0.390|
Other findings:
- At the cut-off level 0.30 the model identifies 68% of defaulters (vs 47% at 0.50), though at the expense

## Results
Test set: 300 loans (stratified 70/30 split). Plots and tables are in the `output` files.

| Metric | Logistic | Lasso |
|---|---|---|
| AUC | 0.752 | 0.746 |
| Gini | 0.504 | 0.493 |
| KS | 0.390 | - |

Other findings:
- At cut-off 0.30 the model flags 68% of defaulters (vs 47% at 0.50), at the cost of more false alarms.
- Lasso keeps 20 of 45 variables with almost the same discriminatory power.
- Calibration: predicted PDs are slightly too low for the safest class and too high for the riskiest (60 loans per class, so noisy).
- Expected loss on the test portfolio: 153,134 on 988,191 of exposure (15.5%).
- Monte Carlo (10,000 runs): mean loss 153,043; 99% VaR 185,482; 99% Expected Shortfall 190,258.

## Link to Basel
- **PD** is the output of the model. Under IRB approaches banks estimate it
  internally, and it must be calibrated to long-run default rates.
- **LGD** is fixed at 45% (Foundation IRB, senior unsecured) as an illustrative
  assumption, not estimated.
- **EAD** is set equal to the loan amount (fully drawn instalment loans).
- Expected Loss = PD x LGD x EAD, computed per loan and aggregated.

## Limitations
- The dataset has a ~30% default rate, much higher than a real bank portfolio,
  so absolute PDs are not realistic. Ranking power (AUC/Gini) is the meaningful
  result; a real model would need recalibration to the portfolio's central tendency.
- Small sample (300 test observations): metrics have wide confidence intervals.
- The Monte Carlo assumes independent defaults (no correlation, no systematic factor), so tail losses are underestimated compared with a real portfolio.
- No out-of-time validation, no stress scenarios, no LGD/EAD models.

## Possible extensions
Stress test (shift key variables and re-estimate PD), IFRS 9 staging,
LGD model, WoE/IV scorecard.

## Data
German Credit (Statlog), UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Download the file `german.data` (not included in this repository).

## How to run
1. Open the R script in RStudio.
2. Click Source. A window opens: select the downloaded `german.data` file.
3. The script prints the results in the console and saves plots and tables in the `output` folder.

Requires R with the packages `pROC` and `glmnet` (installed automatically).
