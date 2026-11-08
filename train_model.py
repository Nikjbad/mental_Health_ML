"""Re-train the model and save Mental_Health_Model.pkl.

Run this if the pickle fails to load (scikit-learn version mismatch):
    python train_model.py
It mirrors mental_Health_Prediction_ML.ipynb (same preprocessing, same
default RandomForest, which scored best on the test set: R2 ~0.88).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Student_Social_Media_And_Mental_Health_Impact.csv"
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

df = pd.read_csv(CSV_PATH).drop_duplicates()
df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)

top_countries = df["Country"].value_counts().index[:10].tolist()
df["Grouped_country"] = df["Country"].where(df["Country"].isin(top_countries), "Other")
print("Top countries:", top_countries)

skewed_col = ["Study_Hours"]
numeric_cols = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks",
                "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
ordinal_col = ["Stress_Level"]
nominal_cols = ["Gender", "Academic_Level", "Most_Used_Platform",
                "Purpose_Of_Use", "Grouped_country"]

X = df[skewed_col + numeric_cols + ordinal_col + nominal_cols]
y = df["Mental_Health_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

preprocessor = ColumnTransformer([
    ("Skewed_Pipeline", Pipeline([("log_transform", FunctionTransformer(np.log1p)),
                                  ("scaler", StandardScaler())]), skewed_col),
    ("Plain_Numeric", Pipeline([("scale", StandardScaler())]), numeric_cols),
    ("Ordinal", Pipeline([("encode", OrdinalEncoder(
        categories=[["Low", "Medium", "High", "Very High"]]))]), ordinal_col),
    ("Normal", Pipeline([("encode", OneHotEncoder(handle_unknown="ignore"))]), nominal_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("random forest", RandomForestRegressor(random_state=42)),
])
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print(f"Test R2 : {r2_score(y_test, pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, pred):.4f}")

joblib.dump(pipeline, MODEL_PATH)
print("Saved ->", MODEL_PATH)
