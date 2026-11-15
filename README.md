# California Housing Price Prediction

**Question:** Can we predict a California district's median house value from census features, and cut RMSE by at least 50% versus a mean baseline?

**Data:** 1990 California housing census data (20,640 rows, 8 features), loaded via scikit-learn. No missing values or duplicates. 992 rows (4.8%) have prices capped at $500k.

## How to run
pip install -r requirements.txt
python project.py

## Method
- Checked missing values, duplicates, outliers; plotted target, location map and correlations.
- Engineered BedroomRatio and RoomsPerPerson.
- 80/20 train/test split (seed 42); outliers clipped at the 99th percentile using training data only.
- Models: mean baseline, linear regression, random forest tuned with 3-fold CV.

## Results (prices in $100,000)
| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Baseline (mean) | 1.145 | 0.906 | -0.000 |
| Linear Regression | 0.664 | 0.480 | 0.663 |
| Random Forest (tuned) | 0.503 | 0.328 | 0.807 |

## Insights
- Median income is the strongest predictor (correlation 0.69); location also matters.
- The tuned random forest cut RMSE by 56% versus the baseline, so the goal was met.
- Error is higher for capped houses (MAE 0.743 vs 0.308), because the model cannot predict above the cap.

## Limitations and next steps
Data is from 1990, prices are capped, rows are district averages, and there are no neighbourhood features. Next: gradient boosting and external data.

##AUTHOR
YASHAS M (EC24I2007)
SUJAN RAM S(EC24I2008)
MUKESH MG (EC24I2020)
