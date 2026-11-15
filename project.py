import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
df = fetch_california_housing(as_frame=True).frame
target = "MedHouseVal"

# ---- Data checks ----
print(df.shape, "missing:", df.isna().sum().sum(), "duplicates:", df.duplicated().sum())
print(df.describe().T)
print("Rows with target capped at 5.0:", (df[target] >= 5.0).sum())  # dataset censors prices at 5.0 (~$500k)

# ---- EDA plots ----
df[target].hist(bins=50); plt.title("House value distribution"); plt.savefig("eda_target.png"); plt.close()
plt.scatter(df.Longitude, df.Latitude, c=df[target], s=2, cmap="viridis"); plt.colorbar()
plt.title("Price by location"); plt.savefig("eda_map.png"); plt.close()
corr = df.corr()[target].drop(target).sort_values(); corr.plot.barh(title="Correlation with price")
plt.tight_layout(); plt.savefig("eda_corr.png"); plt.close()
print("\nCorrelation with price:\n", corr)

# ---- Feature engineering ----
df["BedroomRatio"] = df.AveBedrms / df.AveRooms
df["RoomsPerPerson"] = df.AveRooms / df.AveOccup

# ---- Split (test set untouched until the end) ----
X, y = df.drop(columns=target), df[target]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ---- Outliers: clip extreme values using TRAIN quantiles only (avoids leakage) ----
for c in ["AveRooms", "AveBedrms", "AveOccup", "BedroomRatio", "RoomsPerPerson", "Population"]:
    hi = Xtr[c].quantile(0.99)
    Xtr[c], Xte[c] = Xtr[c].clip(upper=hi), Xte[c].clip(upper=hi)

def report(name, model):
    p = model.predict(Xte)
    r = dict(model=name, RMSE=mean_squared_error(yte, p) ** 0.5, MAE=mean_absolute_error(yte, p), R2=r2_score(yte, p))
    print(r); return r, p

# ---- Models: baseline, linear, tuned random forest (3-fold CV) ----
rows = []
rows.append(report("Baseline (mean)", DummyRegressor().fit(Xtr, ytr))[0])
rows.append(report("Linear Regression", LinearRegression().fit(Xtr, ytr))[0])
gs = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
                  {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_leaf": [1, 3]},
                  cv=3, scoring="neg_root_mean_squared_error").fit(Xtr, ytr)
print("Best RF params:", gs.best_params_, "CV RMSE:", -gs.best_score_)
r, pred = report("Random Forest (tuned)", gs.best_estimator_); rows.append(r)
pd.DataFrame(rows).round(3).to_csv("results.csv", index=False)

# ---- Error analysis ----
res = yte - pred
plt.scatter(pred, res, s=3); plt.axhline(0, c="r"); plt.xlabel("Predicted"); plt.ylabel("Residual")
plt.title("Residuals"); plt.savefig("residuals.png"); plt.close()
print("\nMean abs error for houses at the 5.0 cap:", res[yte >= 5.0].abs().mean(),
      "| others:", res[yte < 5.0].abs().mean())
pd.Series(gs.best_estimator_.feature_importances_, index=X.columns).sort_values().plot.barh(title="RF importance")
plt.tight_layout(); plt.savefig("importance.png")