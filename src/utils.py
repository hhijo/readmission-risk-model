# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




def load_csv(path):
return pd.read_csv(path)




def save_model(model, path):
import joblib
joblib.dump(model, path)




def load_model(path):
import joblib
return joblib.load(path)




def train_val_test_split(df, target_col, test_size=0.15, val_size=0.15, random_state=42):
# If temporal splitting needed, replace with time-based split
train_val, test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
val_relative = val_size / (1 - test_size)
train, val = train_test_split(train_val, test_size=val_relative, stratify=train_val[target_col], random_state=random_state)
return train, val, test
