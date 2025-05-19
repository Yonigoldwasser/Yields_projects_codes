#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# In[11]:


# --- Load your CSV ---
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")  # Replace with your actual path

columns_to_drop = ['state', 'clucalcula','commonland','yield_group']
df = df.drop(columns=columns_to_drop)

# --- Preprocess ---
# Assume first column is field ID, last column is 'yield'
field_column = df.columns[0] # leave out ID, yield, calcul, state, yield group, commonland
yield_column = '2023_yield'

df = df.dropna(subset=[yield_column])
df.set_index(field_column, inplace=True)

X = df.drop(columns=[yield_column])
X = X.select_dtypes(include=[np.number]).fillna(X.mean())
y = df[yield_column]


# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model results dictionary ---
results = {}

# ElasticNet
enet = ElasticNetCV(cv=5, random_state=0)
enet.fit(X_train, y_train)
results['ElasticNet'] = r2_score(y_test, enet.predict(X_test))

# PLS Regression
pls = PLSRegression(n_components=min(10, X.shape[1]))
pls.fit(X_train, y_train)
results['PLSRegression'] = r2_score(y_test, pls.predict(X_test))

# SVR
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10))
svr.fit(X_train, y_train)
results['SVR'] = r2_score(y_test, svr.predict(X_test))

# XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
results['XGBoost'] = r2_score(y_test, xgb.predict(X_test))

# PCA + KMeans
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# --- Summary table ---
results_df = pd.DataFrame(list(results.items()), columns=["Model", "R2 Score"])
results_df['Rank'] = results_df['R2 Score'].rank(ascending=False).astype(int)
results_df.sort_values(by="R2 Score", ascending=False, inplace=True)

print("\n--- Model Performance Summary ---")
print(results_df)

# Optional: Plot feature importances for XGBoost
importances = pd.Series(xgb.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', title='Top 10 XGBoost Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[20]:


# --- Load your CSV ---
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_including_indices_dates_interpolation_v4_20250417_PDreal.csv")  # Replace with your actual path

columns_to_drop = ['state', 'clucalcula','commonland','yield_group']
df = df.drop(columns=columns_to_drop)

# --- Preprocess ---
field_column = df.columns[0]
yield_column = '2023_yield'

df = df.dropna(subset=[yield_column])
df.set_index(field_column, inplace=True)

X = df.drop(columns=[yield_column])
X = X.select_dtypes(include=[np.number])
y = df[yield_column]

# --- Impute missing values ---
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# --- Model results dictionary ---
results = {}

# ElasticNet
print("start ElasticNet")
enet = ElasticNetCV(cv=5, random_state=0)
enet.fit(X_train, y_train)
results['ElasticNet'] = r2_score(y_test, enet.predict(X_test))

# PLS Regression
print("start PLS Regression")
pls = PLSRegression(n_components=min(10, X.shape[1]))
pls.fit(X_train, y_train)
results['PLSRegression'] = r2_score(y_test, pls.predict(X_test))

# SVR
print("start SVR")
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10))
svr.fit(X_train, y_train)
results['SVR'] = r2_score(y_test, svr.predict(X_test))

# XGBoost
print("start XGBoost")
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
results['XGBoost'] = r2_score(y_test, xgb.predict(X_test))

# PCA + KMeans
print("start PCA + KMeans")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_imputed)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# --- Summary table ---
print("start Summary table")
results_df = pd.DataFrame(list(results.items()), columns=["Model", "R2 Score"])
results_df['Rank'] = results_df['R2 Score'].rank(ascending=False).astype(int)
results_df.sort_values(by="R2 Score", ascending=False, inplace=True)

print("\n--- Model Performance Summary ---")
print(results_df)

# --- Optional: Plot top XGBoost features ---
importances = pd.Series(xgb.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', title='Top 10 XGBoost Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[21]:


# --- Load your CSV ---
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_including_indices_dates_interpolation_v4_20250417_PDreal.csv")  # Replace with your actual path

columns_to_drop = ['state', 'clucalcula','commonland','yield_group']
df = df.drop(columns=columns_to_drop)

# --- Preprocess ---
field_column = df.columns[0]
yield_column = '2023_yield'

df = df.dropna(subset=[yield_column])
df.set_index(field_column, inplace=True)

X = df.drop(columns=[yield_column])
X = X.select_dtypes(include=[np.number])
y = df[yield_column]

# --- Impute missing values ---
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# --- Model results dictionary ---
results = {}

# ElasticNet
print("start ElasticNet")
# ElasticNet with StandardScaler in pipeline to avoid convergence issues
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

enet_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNetCV(cv=5, random_state=0, max_iter=10000))
])
enet_pipeline.fit(X_train, y_train)
results['ElasticNet'] = r2_score(y_test, enet_pipeline.predict(X_test))

# PLS Regression
print("start PLS Regression")
pls = PLSRegression(n_components=min(10, X.shape[1]))
pls.fit(X_train, y_train)
results['PLSRegression'] = r2_score(y_test, pls.predict(X_test))

# SVR
print("start SVR")
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10))
svr.fit(X_train, y_train)
results['SVR'] = r2_score(y_test, svr.predict(X_test))

# XGBoost
print("start XGBoost")
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
results['XGBoost'] = r2_score(y_test, xgb.predict(X_test))

# PCA + KMeans
print("start PCA + KMeans")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_imputed)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# --- Summary table ---
print("start Summary table")
results_df = pd.DataFrame(list(results.items()), columns=["Model", "R2 Score"])
results_df['Rank'] = results_df['R2 Score'].rank(ascending=False).astype(int)
results_df.sort_values(by="R2 Score", ascending=False, inplace=True)

print("\n--- Model Performance Summary ---")
print(results_df)

# --- Optional: Plot top XGBoost features ---
importances = pd.Series(xgb.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', title='Top 10 XGBoost Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

enet_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNetCV(cv=5, random_state=0, max_iter=10000))
])
enet_pipeline.fit(X_train, y_train)
results['ElasticNet'] = r2_score(y_test, enet_pipeline.predict(X_test))


# In[ ]:




