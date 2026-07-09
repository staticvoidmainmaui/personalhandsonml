# ML Project Workflow — Housing Notebook Summary

A reusable checklist distilled from `Housing.ipynb`, organized by topic. Use this as a template when starting a new project on a different dataset.

## 1. Setup & Data Acquisition
- Check library versions early (e.g. assert `sklearn` version) to avoid silent API mismatches.
- Write a `fetch_data()` helper that downloads/extracts into a `datasets/<name>/` folder if it doesn't already exist.
- Write a `load_data()` helper that just reads the CSV with `pandas.read_csv`.

## 2. First Look at the Data
- `df.head()`, `df.info()` — check dtypes, non-null counts, spot missing values.
- `df["col"].value_counts()` for categorical columns.
- `df.describe()` for numerical summary stats (mean, std, percentiles).
- `df.hist(bins=50, figsize=(12,8))` — histogram every numeric column at once to spot skew, caps, and scale differences.

## 3. Create a Test Set (before exploring further!)
- Split off a test set immediately to avoid "data snooping bias" — don't let your own analysis of the test data influence modeling choices.
- Simple random split: `sklearn.model_selection.train_test_split(df, test_size=0.2, random_state=42)`.
- If a feature is important and skewed (e.g. income), create a categorical bucket of it (`pd.cut`) and use **Stratified Sampling** (`StratifiedShuffleSplit`) so train/test sets match the population distribution on that feature — more robust than pure random splitting.
- Set a `random_state`/`np.random.seed()` for reproducibility.

## 4. Exploratory Data Analysis (on a copy of the training set only)
- `df.plot(kind="scatter", x=..., y=...)` for geographic/spatial data; use `alpha` to reveal density.
- Bubble/color-coded scatter (`s=size_col`, `c=color_col`, `cmap="jet"`) to visualize 3–4 variables at once.
- `df.corr(numeric_only=True)["target"].sort_values()` — quick linear-correlation ranking against the label.
- `pandas.plotting.scatter_matrix()` on the most-correlated attributes to check pairwise relationships visually.
- Try engineering combination features (e.g. rooms_per_household) — often more predictive than raw counts.

## 5. Prepare the Data for ML
- Separate predictors from labels: `X = df.drop(label, axis=1)`, `y = df[label].copy()`.
- **Missing values**: `sklearn.impute.SimpleImputer(strategy="median")` fit/transform on numeric columns only.
- **Outliers**: optionally detect with `IsolationForest.fit_predict()` and consider dropping.
- **Categorical/text features**:
  - `OrdinalEncoder` if categories have order (rarely) — otherwise avoid, since it implies false ordering.
  - `OneHotEncoder` (or `pd.get_dummies`) is the default for unordered categories.
- **Custom transformers**: subclass `BaseEstimator` + `TransformerMixin` to build reusable feature-engineering steps (e.g. adding ratios) that plug into a pipeline.
- **Pipelines**: chain preprocessing steps with `Pipeline([...])` so `fit_transform` is one call and order is guaranteed.
- **ColumnTransformer**: apply different pipelines to different column subsets (numeric pipeline + categorical encoder) in one object — this is what you `fit_transform` to get your final model-ready matrix.

## 6. Select and Train a Model
- Start simple: `LinearRegression` as a baseline.
- Evaluate with RMSE: `sqrt(mean_squared_error(y_true, y_pred))`.
- If underfitting, try a more powerful model (`DecisionTreeRegressor`, `RandomForestRegressor`) — but watch for overfitting (huge train-score gap vs. cross-validated score).
- **Always cross-validate** instead of trusting a single train-score: `cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)`, then `sqrt(-scores)`. Compare mean and std across models — std tells you how stable the estimate is.
- Save promising models with `joblib.dump()/load()` so you don't retrain from scratch.

## 7. Fine-Tune the Model
- `GridSearchCV(model, param_grid, cv=5, scoring=...)` to systematically search hyperparameters; inspect `.best_params_`, `.best_estimator_`, `.cv_results_`.
- For large search spaces, prefer `RandomizedSearchCV` (samples a fixed number of combinations instead of exhaustive grid).
- Inspect `feature_importances_` on tree-based models to see which features matter — useful for dropping noise features or explaining the model.

## 8. Evaluate on the Test Set — once
- Transform the untouched test set through the *same* fitted pipeline (`pipeline.transform`, not `fit_transform`).
- Compute final RMSE on test predictions.
- Report a confidence interval on the generalization error (`scipy.stats.t.interval`) instead of a single point estimate.
- Golden rule: **never** tune hyperparameters against test-set performance — that data is used exactly once, at the end.

## 9. Launch, Monitor, Maintain
- Monitor live performance in production and set alerts for degradation.
- Periodically sample and manually review predictions — automated metrics alone can miss real-world failure modes.
- Monitor input data quality/drift — "fresh input = fresh output"; stale or shifted input distributions degrade models silently.
- Retrain regularly and keep versioned model/metric history to support rollback.

---

## Template Checklist for a New Dataset
1. [ ] Fetch/load data into `datasets/<name>/`.
2. [ ] `head()`, `info()`, `describe()`, `value_counts()`, `hist()`.
3. [ ] Split off test set (stratified on the most important skewed feature if applicable).
4. [ ] EDA on training copy: correlations, scatter plots, feature engineering ideas.
5. [ ] Build a `ColumnTransformer` pipeline: impute + scale numeric, encode categorical, add engineered features via a custom transformer.
6. [ ] Baseline model → cross-validate → try 2–3 stronger models → cross-validate each.
7. [ ] `GridSearchCV`/`RandomizedSearchCV` on the best model.
8. [ ] Evaluate once on the test set, report RMSE + confidence interval.
9. [ ] Save the final pipeline+model with `joblib`; plan for monitoring/retraining.
