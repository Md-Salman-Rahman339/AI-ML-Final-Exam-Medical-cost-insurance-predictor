import pandas as pd
import numpy as np
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("insurance.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

df['bmi_category'] = pd.cut(
    df['bmi'],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=['underweight', 'normal', 'overweight', 'obese']
)

df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 35, 50, 100],
    labels=['teen', 'young', 'adult', 'senior']
)

df['family_size'] = df['children'] + 1

print("\nFeature Engineering Applied")

X = df.drop('charges', axis=1)
y = df['charges']

print(f"\nFeatures shape: {X.shape}")


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

print("\nNumerical Columns:", list(numeric_features))
print("\nCategorical Columns:", list(categorical_features))


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])


rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

print("\nRandom Forest selected because it handles non-linear relationships and tabular data effectively.")


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
train_pred = pipeline.predict(X_train)


print("\nTraining Performance:")
print(f"R2 Score: {r2_score(y_train, train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_train, train_pred):.4f}")



param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.4f}")


best_model = grid_search.best_estimator_
print("\nBest Model:\n", best_model)

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")



with open("insurance_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns, f)

print("\n Model saved as insurance_model.pkl")
print(" Features saved as feature_names.pkl")