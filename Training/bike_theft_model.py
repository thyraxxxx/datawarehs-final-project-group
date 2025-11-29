
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 50)

# ---------------------------------------------------
# 1) Load Dataset
# ---------------------------------------------------
df = pd.read_csv("bike_thefts.csv")

print(df.columns.values)
print(df.head())
print(df.info())

print("Missing values:")
print(df.isnull().sum())

# ---------------------------------------------------
# 2) Select Columns for Training
# ---------------------------------------------------
FEATURES = [
    'OCC_DATE', 'REPORT_DATE', 'NEIGHBOURHOOD_158', 'BIKE_MODEL',
    'BIKE_TYPE', 'PRIMARY_OFFENCE', 'BIKE_COST', 'PREMISES_TYPE',
    'LOCATION_TYPE', 'EVENT_UNIQUE_ID', 'BIKE_SPEED', 'NEIGHBOURHOOD_140',
    'BIKE_COLOUR'
]

TARGET = "STATUS"

df = df[FEATURES + [TARGET]]
print(df.head())

# ---------------------------------------------------
# 3) Split Train/Test Using Stratified Sampling
# ---------------------------------------------------
df = df.dropna(subset=[TARGET])  # drop target missing

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

for train_index, test_index in sss.split(df, df[TARGET]):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]

print("Train size:", train_set.shape)
print("Test size:", test_set.shape)
print(train_set[TARGET].value_counts(normalize=True))
print(test_set[TARGET].value_counts(normalize=True))

# ---------------------------------------------------
# 4) FEATURE ENGINEERING — DATE PARSING
# ---------------------------------------------------

def add_date_features(df):
    df = df.copy()
    
    df['OCC_DATE'] = pd.to_datetime(df['OCC_DATE'], errors='coerce')
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce')

    # OCC fields
    df['OCC_YEAR'] = df['OCC_DATE'].dt.year
    df['OCC_MONTH'] = df['OCC_DATE'].dt.month
    df['OCC_DAY'] = df['OCC_DATE'].dt.day
    df['OCC_HOUR'] = df['OCC_DATE'].dt.hour
    df['OCC_DOW'] = df['OCC_DATE'].dt.dayofweek
    df['OCC_DOY'] = df['OCC_DATE'].dt.dayofyear

    # REPORT fields
    df['REPORT_MONTH'] = df['REPORT_DATE'].dt.month
    df['REPORT_DAY'] = df['REPORT_DATE'].dt.day
    df['REPORT_HOUR'] = df['REPORT_DATE'].dt.hour
    df['REPORT_DOW'] = df['REPORT_DATE'].dt.dayofweek
    df['REPORT_DOY'] = df['REPORT_DATE'].dt.dayofyear

    return df

train_set = add_date_features(train_set)
test_set = add_date_features(test_set)

# ---------------------------------------------------
# 5) Separate Features/Labels
# ---------------------------------------------------
y_train = train_set[TARGET]
X_train = train_set.drop(columns=[TARGET])

# Identify categorical vs numeric columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------
# 6) Preprocessing Pipelines
# ---------------------------------------------------
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("categorical", cat_pipeline, categorical_cols),
    ("numeric", num_pipeline, numeric_cols)
])

# Fit the pipeline
X_train_prepared = preprocessor.fit_transform(X_train)

# ---------------------------------------------------
# 7) Logistic Regression + Cross Validation
# ---------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

lr = LogisticRegression(max_iter=2000, solver='lbfgs')

cv = KFold(n_splits=10, shuffle=True, random_state=1)
score = np.mean(cross_val_score(lr, X_train_prepared, y_train, cv=cv, scoring='accuracy'))

print("Cross-val accuracy:", score)

# ---------------------------------------------------
# 8) Hyperparameter Tuning
# ---------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid = {
    "C": randint(1, 20),
    "penalty": ["l2"]
}

clf = RandomizedSearchCV(
    estimator=lr,
    param_distributions=param_grid,
    n_iter=50,
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train_prepared, y_train)

print("Best params:", clf.best_params_)
best_model = clf.best_estimator_

# ---------------------------------------------------
# 9) Dump model + pipeline + feature order
# ---------------------------------------------------
import joblib

joblib.dump(best_model, "model.pkl")
joblib.dump(preprocessor, "pipeline.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")
joblib.dump(numeric_cols, "numeric_cols.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

print("Saved model, pipeline, and feature files!")
