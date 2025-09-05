import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
df = pd.read_csv(path)
return df




def preprocess(df: pd.DataFrame, target: str = 'charges') -> tuple:
# Basic cleaning
df = df.copy()
# Drop rows with missing target
df = df.dropna(subset=[target])


# Fill numeric missing with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != target]
for c in num_cols:
df[c] = df[c].fillna(df[c].median())


# Fill categorical missing with mode
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_cols:
df[c] = df[c].fillna(df[c].mode().iloc[0])


# Feature engineering: BMI bucket
if 'bmi' in df.columns:
df['bmi_cat'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100], labels=['under','normal','over','obese'])


# Train-test split
if 'smoker' in df.columns:
strat = df['smoker']
else:
strat = None


X = df.drop(columns=[target])
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
return X_train, X_test, y_train, y_test
