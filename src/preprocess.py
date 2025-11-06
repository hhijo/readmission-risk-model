# preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




def basic_preprocess(df, target_col="readmit_30"):
# Example pipeline — adapt to your hospital's schema
df = df.copy()


# Create target if not present (example placeholder)
if target_col not in df.columns:
raise ValueError(f"Target column {target_col} not found in dataframe")


# Example engineered features
df["num_prior_adm_90d"] = df["prior_admissions_90d"].fillna(0)
df["age_bins"] = pd.cut(df["age"], bins=[0,18,40,65,200], labels=["child","young","adult","elderly"])


# Select features
numeric_cols = [c for c in df.columns if df[c].dtype in ["int64","float64"] and c != target_col]
cat_cols = [c for c in df.columns if df[c].dtype == "object"]


# Impute numeric
num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])


# One-hot encode small cardinality categoricals
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# Scale numeric
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
return X, y
