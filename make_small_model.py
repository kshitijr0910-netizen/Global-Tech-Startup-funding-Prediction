import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("global_tech_startups_2026.csv")

X = df.drop(columns=["Total_Funding_USD_Millions"])
y = df["Total_Funding_USD_Millions"]

num_cols = [
    "Valuation_USD_Millions",
    "Runway_Months_2024",
    "Peak_Headcount_2023",
    "Layoffs_2024_2025",
    "Current_Headcount_2026"
]

cat_cols = [
    "Domain",
    "Country",
    "City",
    "Funding_Stage"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regression", RandomForestRegressor(
            n_estimators=75,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

print("Training model...")

model.fit(X, y)

print("Saving model...")

joblib.dump(
    model,
    "funding_usd_pred_small.joblib",
    compress=9
)

print("DONE! Small model created.")
