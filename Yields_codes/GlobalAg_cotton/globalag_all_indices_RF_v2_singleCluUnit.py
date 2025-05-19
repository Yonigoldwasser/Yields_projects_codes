#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For enhanced plotting
from joblib import Parallel, delayed # For parallel processing


# # stack column by index and date (without year)
# 

# In[26]:


import pandas as pd
import re
# Load the full DataFrame
df = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit.csv")  # Update this path

# Define ID column
id_col = 'PWId'

# --- DAILY COLUMNS (e.g., 20220723_NDVI) ---
daily_cols = [col for col in df.columns if col.count('_') == 1 and col.split('_')[0].isdigit()]
df_daily = df[[id_col] + daily_cols].copy()

df_daily_long = df_daily.melt(id_vars=id_col, var_name='col', value_name='value')
df_daily_long['Year'] = df_daily_long['col'].str[:4]
df_daily_long['new_col'] = df_daily_long['col'].str[4:8] + '_' + df_daily_long['col'].str.extract(r'_(\w+)$')[0]
df_daily_long['UID'] = df_daily_long[id_col].astype(str) + '_' + df_daily_long['Year']
daily_pivot = df_daily_long.pivot(index='UID', columns='new_col', values='value')

# --- AGGREGATED COLUMNS (e.g., RedEdge1_2022_max_August) ---
agg_cols = [col for col in df.columns if col.count('_') >= 3 and not col.split('_')[0].isdigit()]
df_agg = df[[id_col] + agg_cols].copy()

df_agg_long = df_agg.melt(id_vars=id_col, var_name='col', value_name='value')
df_agg_long[['Index', 'Year', 'Metric', 'Month']] = df_agg_long['col'].str.extract(r'(\w+)_(\d{4})_(\w+)_(\w+)')
df_agg_long['new_col'] = df_agg_long['Metric'] + '_' + df_agg_long['Month'] + '_' + df_agg_long['Index']
df_agg_long['UID'] = df_agg_long[id_col].astype(str) + '_' + df_agg_long['Year']
agg_pivot = df_agg_long.pivot(index='UID', columns='new_col', values='value')

# --- SIMPLE YEARLY COLUMNS (e.g., 2019_yield) ---
simple_cols = [
    col for col in df.columns
    if re.match(r'^\d{4}_.+', col) and not re.match(r'^\d{8}_', col)  # year prefix, but not daily
]
print('simple_cols', simple_cols)

df_simple = df[[id_col] + simple_cols].copy()

df_simple_long = df_simple.melt(id_vars=id_col, var_name='col', value_name='value')
df_simple_long['Year'] = df_simple_long['col'].str.extract(r'(\d{4})')
df_simple_long['new_col'] = (
    df_simple_long['col']
    .str.replace(r'^\d{4}_+', '', regex=True)  # remove year and extra underscores
    .str.strip()
)
df_simple_long['UID'] = df_simple_long[id_col].astype(str) + '_' + df_simple_long['Year']
simple_pivot = df_simple_long.pivot(index='UID', columns='new_col', values='value')

# --- MOVE 'yield' COLUMN TO END ---
yield_col = None
if 'yield' in simple_pivot.columns:
    yield_col = simple_pivot.pop('yield')

# --- COMBINE ALL PIVOTED TABLES ---
final_df = pd.concat([daily_pivot, agg_pivot, simple_pivot], axis=1)

# Re-attach 'yield' at the end
if yield_col is not None:
    final_df['yield'] = yield_col

# --- SPLIT UID BACK INTO PWId AND Year ---
final_df = final_df.reset_index()
final_df[['PWId', 'Year']] = final_df['UID'].str.split('_', expand=True)
final_df = final_df.drop(columns='UID')
final_df
# --- (Optional) Save to Excel or CSV ---
# final_df.to_csv("reshaped_globalag_data.csv", index=False)
# final_df.to_excel("reshaped_globalag_data.xlsx", index=False)


# In[27]:


final_df = final_df[final_df['yield'].notna() & (final_df['yield'] != 0)]

# final_df.to_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit_reshaped.csv", index=False)
final_df


# In[35]:


columns_list = final_df.columns.tolist()
print(columns_list)
if '_yie_yield' in final_df.columns:
    final_df = final_df.drop(columns='_yie_yield')
print('')
print(final_df.columns.tolist())


# # RF with acres and annual production threshold
# 

# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print("ok")
# --- Data Preparation ---
def prepare_data(df):
    if 'yield' not in df.columns:
        raise ValueError("Expected a column named 'yield' in the DataFrame.")
    
    if 'Year' not in df.columns:
        raise ValueError("Expected a column named 'Year' in the DataFrame.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['yield', 'Year', 'PWId']]

    df = df[df['yield'].notna() & (df['yield'] > 0)].copy()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # Split features and labels
    X = df[feature_cols]
    y = df['yield']
    years = df['Year']
    
    # Split with index tracking for years
    X_train, X_test, y_train, y_test, years_train, years_test = train_test_split(
        X, y, years, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_cols, years_test

# --- Model Training ---
def train_random_forest(X_train, y_train, param_grid=None):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    if param_grid:
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best hyperparameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        rf.fit(X_train, y_train)
        return rf

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r_squared:.2f}, MAE: {mae:.2f}")
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R-squared': r_squared,
        'MAE': mae
    }

# --- Feature Importance ---
def get_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:top_n]]

    print("Top Feature Importances:")
    for name, val in top_features:
        print(f"{name}: {val:.4f}")
    return top_features

def save_feature_importance_csv(top_features, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df_feat = pd.DataFrame(top_features, columns=["Feature", "Importance"])
    path = os.path.join(output_folder, 'top_features.csv')
    df_feat.to_csv(path, index=False)
    print(f"Feature importances saved to {path}")

# --- Plotting ---
def plot_predictions(y_test, y_pred, r_squared, output_folder, years):
    plt.figure(figsize=(8, 6))
    plot_df = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred,
        'Year': years
    })

    sns.scatterplot(data=plot_df, x='True', y='Predicted', hue='Year', palette='tab10', alpha=1)
    sns.regplot(data=plot_df, x='True', y='Predicted', scatter=False, color='black')
    plt.xlabel("True Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Predicted vs. True Yield")
    plt.text(0.05, 0.95, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes)
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, 'prediction_vs_actual_by_year.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to {path}")

# --- Main Execution ---
if __name__ == "__main__":
    # Load data
    df = final_df

    # Make sure 'Year' column exists or add it manually here if needed:
    # Example: df['Year'] = 2021  # If all rows are from 2021

    # Set output folder
    output_dir = r"F:\globalag\clus_sandwich_test\v2_rf_outputs_singleCluUnit"

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names, test_years = prepare_data(df)

    # Optional hyperparameter tuning (can enable later)
    param_grid = None  # or define a grid like {'n_estimators': [100, 200], ...}

    # Train model
    model = train_random_forest(X_train, y_train, param_grid)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importances
    top_features = get_feature_importance(model, feature_names, top_n=10)
    save_feature_importance_csv(top_features, output_dir)

    # Plot predictions by year
    plot_predictions(y_test, model.predict(X_test), metrics['R-squared'], output_dir, test_years)


# # RF with each year as test

# In[37]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("ok")

# --- Data Preparation ---
def prepare_data(df):
    if 'yield' not in df.columns:
        raise ValueError("Expected a column named 'yield' in the DataFrame.")
    
    if 'Year' not in df.columns:
        raise ValueError("Expected a column named 'Year' in the DataFrame.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['yield', 'Year', 'PWId']]

    df = df[df['yield'].notna() & (df['yield'] > 0)].copy()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    X = df[feature_cols]
    y = df['yield']
    years = df['Year']

    return X, y, feature_cols, years

# --- Model Training ---
def train_random_forest(X_train, y_train, param_grid=None):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    if param_grid:
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best hyperparameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        rf.fit(X_train, y_train)
        return rf

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r_squared:.2f}, MAE: {mae:.2f}")
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R-squared': r_squared,
        'MAE': mae,
        'y_true': y_test,
        'y_pred': y_pred
    }

# --- Feature Importance ---
def get_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:top_n]]

    print("Top Feature Importances:")
    for name, val in top_features:
        print(f"{name}: {val:.4f}")
    return top_features

def save_feature_importance_csv(top_features, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df_feat = pd.DataFrame(top_features, columns=["Feature", "Importance"])
    path = os.path.join(output_folder, 'top_features.csv')
    df_feat.to_csv(path, index=False)
    print(f"Feature importances saved to {path}")

# --- Plotting ---
def plot_predictions(y_test, y_pred, r_squared, output_folder, years):
    plt.figure(figsize=(8, 6))
    plot_df = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred,
        'Year': years
    })

    sns.scatterplot(data=plot_df, x='True', y='Predicted', hue='Year', palette='tab10', alpha=1)
    sns.regplot(data=plot_df, x='True', y='Predicted', scatter=False, color='black')
    plt.xlabel("True Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Predicted vs. True Yield")
    plt.text(0.05, 0.95, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes)
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, 'prediction_vs_actual_by_year.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to {path}")

# --- Main Execution ---
if __name__ == "__main__":
    # Load your final DataFrame
    df = final_df  # Replace this with actual data loading if needed

    output_dir = r"F:\globalag\clus_sandwich_test\v2_rf_outputs_singleCluUnit_years_test"
    os.makedirs(output_dir, exist_ok=True)

    unique_years = df['Year'].dropna().unique()
    summary_metrics = []

    for test_year in unique_years:
        print(f"\n--- Testing on year: {test_year} ---")

        train_df = df[df['Year'] != test_year].copy()
        test_df = df[df['Year'] == test_year].copy()

        if len(test_df) < 5:
            print(f"Skipping year {test_year} (too few samples)")
            continue

        # Prepare
        X_train, y_train, feature_names, _ = prepare_data(train_df)
        X_test, y_test, _, test_years = prepare_data(test_df)

        # Train
        model = train_random_forest(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Year'] = test_year
        summary_metrics.append(metrics)

        # Save Feature Importance
        top_features = get_feature_importance(model, feature_names, top_n=10)
        year_output_dir = os.path.join(output_dir, f"test_year_{test_year}")
        save_feature_importance_csv(top_features, year_output_dir)

        # Plot
        plot_predictions(y_test, metrics['y_pred'], metrics['R-squared'], year_output_dir, test_years)

    # Save summary metrics
    summary_df = pd.DataFrame(summary_metrics)
    summary_path = os.path.join(output_dir, "yearly_metrics_summary.csv")
    summary_df.drop(columns=['y_true', 'y_pred']).to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to {summary_path}")


# # stats and histogram

# In[39]:


data_for_irr = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit.csv")  # Update this path
data_for_irr


# In[46]:


stats_df = final_df.copy()  # optional: prevent modifying final_df directly

# Ensure both are string type and trimmed
stats_df['PWId'] = stats_df['PWId'].astype(str).str.strip()
data_for_irr['PWId'] = data_for_irr['PWId'].astype(str).str.strip()

# Create mapping
irr_map = dict(zip(data_for_irr['PWId'], data_for_irr['PracticeCode']))

# Apply mapping
stats_df['PracticeCode'] = stats_df['PWId'].map(irr_map)
stats_df


# In[47]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming final_df is already defined
# 1. Count rows per year
year_counts = stats_df['Year'].value_counts().sort_index()
print("Number of rows per year:")
print(year_counts)

# 2. Plot histogram of 'yield' per year
unique_years = sorted(stats_df['Year'].dropna().unique())
plt.figure(figsize=(15, 4 * len(unique_years)))

for i, year in enumerate(unique_years, start=1):
    data = final_df[final_df['Year'] == year]['yield'].dropna()
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()

    plt.subplot(len(unique_years), 1, i)
    sns.histplot(data, bins=30, kde=True, color='skyblue')
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
    plt.title(f'Yield Histogram - {year} (n={len(data)})')
    plt.xlabel("Yield")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Count rows per Year and PracticeCode ---
group_counts = stats_df.groupby(['Year', 'PracticeCode']).size().unstack(fill_value=0)
print("Row counts by Year and PracticeCode:")
print(group_counts)

# --- Plot histograms ---
unique_years = sorted(stats_df['Year'].dropna().unique())
practice_codes = stats_df['PracticeCode'].dropna().unique()

plt.figure(figsize=(15, 5 * len(unique_years)))

plot_idx = 1
for year in unique_years:
    for pcode in practice_codes:
        subset = stats_df[(stats_df['Year'] == year) & (stats_df['PracticeCode'] == pcode)]
        data = subset['yield'].dropna()
        if len(data) == 0:
            continue

        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        plt.subplot(len(unique_years), len(practice_codes), plot_idx)
        sns.histplot(data, bins=30, kde=True, color='skyblue')
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
        plt.title(f'Year: {year}, Practice: {pcode} (n={len(data)})')
        plt.xlabel("Yield")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        plot_idx += 1

plt.tight_layout()
plt.show()


# # RF with each year and practice code as test

# In[50]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("ok")

# --- Data Preparation ---
def prepare_data(df):
    if 'yield' not in df.columns or 'Year' not in df.columns:
        raise ValueError("Expected 'yield' and 'Year' columns in the DataFrame.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['yield', 'Year', 'PWId']]

    df = df[df['yield'].notna() & (df['yield'] > 0)].copy()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    X = df[feature_cols]
    y = df['yield']
    years = df['Year']

    return X, y, feature_cols, years

# --- Model Training ---
def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"    MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r_squared:.2f}, MAE: {mae:.2f}")
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R-squared': r_squared,
        'MAE': mae,
        'y_true': y_test,
        'y_pred': y_pred
    }

# --- Feature Importance ---
def get_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:top_n]]

    print("    Top Feature Importances:")
    for name, val in top_features:
        print(f"        {name}: {val:.4f}")
    return top_features

# --- Plotting ---
def plot_predictions(y_test, y_pred, r_squared, years):
    plt.figure(figsize=(8, 6))
    plot_df = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred,
        'Year': years
    })

    sns.scatterplot(data=plot_df, x='True', y='Predicted', hue='Year', palette='tab10', alpha=1)
    sns.regplot(data=plot_df, x='True', y='Predicted', scatter=False, color='black')
    plt.xlabel("True Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Predicted vs. True Yield (R² = {r_squared:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    df = stats_df  # Replace with your DataFrame

    all_metrics = []

    for practice_code in df['PracticeCode'].dropna().unique():
        print(f"\n=== PracticeCode: {practice_code} ===")
        df_group = df[df['PracticeCode'] == practice_code].copy()
        unique_years = df_group['Year'].dropna().unique()

        for test_year in unique_years:
            print(f"\n  --- Testing on Year: {test_year} ---")
            train_df = df_group[df_group['Year'] != test_year]
            test_df = df_group[df_group['Year'] == test_year]

            if len(test_df) < 5 or len(train_df) < 5:
                print("    Skipped (too few samples).")
                continue

            X_train, y_train, feature_names, _ = prepare_data(train_df)
            X_test, y_test, _, test_years = prepare_data(test_df)

            model = train_random_forest(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            metrics['PracticeCode'] = practice_code
            metrics['Year'] = test_year
            all_metrics.append(metrics)

            get_feature_importance(model, feature_names, top_n=10)
            plot_predictions(y_test, metrics['y_pred'], metrics['R-squared'], test_years)

    # Print summary table
    print("\n=== Summary of All Results ===")
    summary_df = pd.DataFrame(all_metrics)
    display_columns = ['PracticeCode', 'Year', 'R-squared', 'RMSE', 'MAE']
    print(summary_df[display_columns].to_string(index=False))


# In[1]:


summary_df


# In[ ]:




