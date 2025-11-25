# -*- coding: utf-8 -*-
"""
Bicycle Theft Project - Person 2 (UPDATED SIMPLE VERSION)
Data cleaning + label encoding + scaling + feature selection + split + imbalance handling
Student: Paule Leslie Stella Kwate
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression

# Optional (for imbalance handling)
try:
    from imblearn.over_sampling import SMOTE
    imblearn_available = True
except Exception:
    imblearn_available = False
    print("imblearn not installed -> SMOTE will be skipped if needed.")

pd.set_option("display.max_columns", None)

# =========================
# 1) LOAD RAW DATA
# =========================
path = r"C:\Users\stell\OneDrive\Bureau\SEMESTER 6\Data Warehouse"
filename = "bicycle_thefts.csv"
fullpath = os.path.join(path, filename)

df = pd.read_csv(fullpath)
print("Loaded:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# =========================
# 2) CREATE TARGET (returned/not returned)
# =========================
possible_target_cols = ["STATUS", "Bike_Status", "BIKE_STATUS", "RETURNED", "RECOVERED"]
target_col = None

for c in possible_target_cols:
    if c in df.columns:
        target_col = c
        break

if target_col is None:
    raise ValueError("Target column not found. Set target_col manually.")

print("Target column detected:", target_col)

# Encode target into binary 1/0
if df[target_col].dtype == "object":
    y = df[target_col].astype(str).str.upper()
    y = y.str.contains("RECOVER|RETURN", regex=True).astype(int)
else:
    y = df[target_col].astype(int)

df["TARGET_RETURNED"] = y


# =========================
# 3) DROP HIGH-UNIQUE ID COLUMNS
# =========================
drop_cols = []
for col in df.columns:
    if col == "TARGET_RETURNED":
        continue
    uniq_ratio = df[col].nunique(dropna=False) / len(df)
    if uniq_ratio > 0.95:
        drop_cols.append(col)

print("Dropping ID-like columns:", drop_cols)
df = df.drop(columns=drop_cols)

# Remove original text target
if target_col in df.columns and target_col != "TARGET_RETURNED":
    df = df.drop(columns=[target_col])


# =========================
# 3b) DROP VERY LARGE TEXT COLUMNS (HIGH CARDINALITY)
# (To avoid one-hot memory explosion)
# =========================
# These names may vary by dataset version, so we drop only if they exist.
high_card_cols = [
    "LOCATION", "LOCN_DETAIL", "STREET1", "STREET2",
    "INTERSECTION", "BIKE_SERIAL_NUMBER", "BIKE_SERIAL_NO",
    "BIKE_MODEL", "BIKE_MAKE", "DIVISION",
    "LATITUDE", "LONGITUDE", "X", "Y"
]

cols_to_drop = [c for c in high_card_cols if c in df.columns]
print("Dropping high-cardinality columns:", cols_to_drop)
df = df.drop(columns=cols_to_drop)


# =========================
# 4) HANDLE MISSING VALUES
# =========================
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

if "TARGET_RETURNED" in numeric_cols:
    numeric_cols.remove("TARGET_RETURNED")

# Fill numeric missing with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing after cleaning:\n", df.isnull().sum().head())


# =========================
# 5) LABEL ENCODE CATEGORICALS (SIMPLE + MEMORY SAFE)
# =========================
# Convert each categorical column into numeric codes
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("After label encoding:", df.shape)


# =========================
# 6) STANDARDIZE FEATURES
# =========================
# Scale ALL predictors so logistic regression behaves well
scaler = StandardScaler()

predictor_cols = [c for c in df.columns if c != "TARGET_RETURNED"]
df[predictor_cols] = scaler.fit_transform(df[predictor_cols])


# =========================
# 7) BUILD X, Y
# =========================
X = df.drop(columns=["TARGET_RETURNED"])
Y = df["TARGET_RETURNED"]

print("X shape:", X.shape)
print("Y distribution:\n", Y.value_counts(normalize=True))


# =========================
# 8) FEATURE SELECTION
# =========================

# A) SelectKBest
k = min(15, X.shape[1])
skb = SelectKBest(score_func=mutual_info_classif, k=k)
skb.fit(X, Y)
selected_kbest = X.columns[skb.get_support()].tolist()
print("\nSelectKBest top features:")
print(selected_kbest)

# B) RFE
lr = LogisticRegression(max_iter=2000, solver="lbfgs")
rfe = RFE(estimator=lr, n_features_to_select=k)
rfe.fit(X, Y)
selected_rfe = X.columns[rfe.support_].tolist()
print("\nRFE top features:")
print(selected_rfe)

# Final chosen features (UNION)
final_features = list(set(selected_kbest).union(selected_rfe))
print("\nFinal chosen features:", final_features)

X_final = X[final_features]


# =========================
# 9) TRAIN/TEST SPLIT
# =========================
trainX, testX, trainY, testY = train_test_split(
    X_final, Y, test_size=0.2, random_state=42, stratify=Y
)

print("Train shape:", trainX.shape, " | Test shape:", testX.shape)


# =========================
# 10) HANDLE IMBALANCE 
# =========================
minority_ratio = trainY.value_counts(normalize=True).min()
print("Minority ratio:", minority_ratio)

if minority_ratio < 0.40 and imblearn_available:
    print("Applying SMOTE oversampling...")
    sm = SMOTE(random_state=42)
    trainX, trainY = sm.fit_resample(trainX, trainY)
    print("After SMOTE:\n", trainY.value_counts())
else:
    print("No SMOTE applied. If imbalanced, use class_weight='balanced' later.")

