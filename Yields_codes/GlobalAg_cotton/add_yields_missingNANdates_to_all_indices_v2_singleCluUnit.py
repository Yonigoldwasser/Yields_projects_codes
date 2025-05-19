#!/usr/bin/env python
# coding: utf-8

# # data preparing

# In[12]:


import geopandas as gpd
import pandas as pd
import ast
import numpy as np
from collections import defaultdict
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from functools import reduce


# ## functions

# In[2]:


def plot_scatter_trend(df, plots_info, output_folder):
    """
    Plots scatter plots with trendlines for specified column pairs in a dataframe and returns a DataFrame
    containing analysis results.

    Parameters:
    - df: DataFrame containing the data.
    - plots_info: List of dictionaries, each containing:
        'x_col': Name of the column for x-axis.
        'y_col': Name of the column for y-axis.
        'color': Color of the scatter plot points.
        'xlabel': Label for the x-axis.
        'ylabel': Label for the y-axis.
        'title': Title of the plot.
        
    Returns:
    - A DataFrame with 'X Column', 'Y Column', 'R² Value', and 'N Observations'.
    """
    results = []  # List to store result dictionaries
    
    # Create subplots
    if len(plots_info) > 1:
        fig, axes = plt.subplots(1, len(plots_info), figsize=(16, 6))
        if len(plots_info) == 1:
            axes = [axes]  # Ensure axes is iterable
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]  # Wrap the single ax in a list for uniform handling
    
    for ax, info in zip(axes, plots_info):
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(df[info['x_col']], df[info['y_col']])
        
        # Scatter plot
        ax.scatter(df[info['x_col']], df[info['y_col']], color=info['color'], label='Data')
        
        # Trendline plot
        ax.plot(df[info['x_col']], slope * df[info['x_col']] + intercept, color='red', label=f'Trendline (R² = {r_value**2:.2f})')
        
        # Set labels and title
        ax.set_xlabel(info['xlabel'])
        ax.set_ylabel(info['ylabel'])
        title = info['title']
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Append result dictionary to results list
        results.append({
            'X Column': info['x_col'],
            'Y Column': info['y_col'],
            'R² Value': r_value**2,
            'N Observations': len(df[info['x_col']])
        })
    
    plot_filename = os.path.join(output_folder, f"{title.replace(' ', '_').replace('.', '')}.png")
    plt.savefig(plot_filename)
    plt.tight_layout()
#     plt.show()
    plt.close()
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_scatter_trend_dfs(dfs, plots_info, output_folder):
    results = []
    for df in dfs:  # Loop through each DataFrame in the list
        # Extract DataFrame name from the second column, if exists and if naming convention applies
        if df.columns.size > 1:
            second_col_name = df.columns[1]
            df_name = second_col_name.split('_')[1] if '_' in second_col_name else 'DataFrame'
        else:
            df_name = 'Unknown DataFrame'
        

        for config in plots_config:  # Access each plot configuration
            print(config)
            # Extract configuration details
            x_col = f"{config['x_col']}_{df_name}"
            y_col = config['y_col']
            color = config['color']
            xlabel = config['xlabel']
            ylabel = config['ylabel']
            title = f"{df_name} - {config['title']}"

                
            # Drop rows with NaN in x_col or y_col to ensure accurate plotting and analysis
            clean_df = df.dropna(subset=[x_col, y_col])
            num_observations = len(clean_df)  # Number of observations

            # Perform regression analysis on the entire DataFrame
            X = sm.add_constant(clean_df[x_col], has_constant='add')  # Add a constant for the intercept
            model = sm.OLS(clean_df[y_col], X, missing='drop').fit()  # Fit the model

            # Create scatter plot with regression line and formula
            plt.figure(figsize=(10, 6))
            plt.scatter(clean_df[x_col], clean_df[y_col], color=color, alpha=0.5)
            plt.plot(clean_df[x_col], model.predict(X), color='red', label=f'Trendline R² = {model.rsquared:.2f}\ny = {model.params[1]:.2f}x + {model.params[0]:.2f}')  # Plot the regression line
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            title = f"{title} (N={num_observations})"
            
            plt.title(title)
            plt.legend()
            plt.grid(True)
            
            plot_filename = os.path.join(output_folder, f"{title.replace(' ', '_').replace('.', '')}.png")
            if not os.path.exists(output_folder):  # Check if the directory exists
                os.makedirs(output_folder)  # Create the directory if it does not exist
            try:
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
            except Exception as e:
                print(f"Failed to save plot: {plot_filename}")
                traceback.print_exc()
            # Show plot
#             plt.show()
            plt.close()

            # Append results for summary
            results.append({
                'DataFrame': df_name,
                'X_column': x_col,
                'R_squared': model.rsquared,
                'N_Observations': num_observations
            })

    # Convert results to a DataFrame
    plot_scatter_trend = pd.DataFrame(results)
    return plot_scatter_trend 
    
def plot_and_collect_regression_results_state(data, plot_configs, output_folder):
    results = []  # List to store results for each state and configuration
    
    for config in plot_configs:
        # Filter data for states to create separate plots for each state
        states = data['state'].unique()
        for state in states:
            state_data = data[data['state'] == state].dropna(subset=[config['x_col'], config['y_col']])
            print('state_data', state_data)
            # Ensure there are enough data points
            if len(state_data) > 1:  # More than one data point is required
                # Fit a regression model
                X = sm.add_constant(state_data[config['x_col']])  # adding a constant
                model = sm.OLS(state_data[config['y_col']], X).fit()
                
                if len(model.params) > 1:  # Check if both parameters are available
                    # Extracting R-squared and formula
                    r_squared = model.rsquared
                    params = model.params
                    label = f"y = {params[1]:.2f}x + {params[0]:.2f}\nR² = {r_squared:.3f}"
                    
                    # Create scatter plot with a regression line
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x=config['x_col'], y=config['y_col'], data=state_data, color=config['color'], scatter_kws={'alpha':0.5})
                    
                    # Setting plot labels and titles
                    plt.xlabel(config['xlabel'])
                    plt.ylabel(config['ylabel'])
                    # Add number of observations to the title
                    title = f"{config['title']} in {state} (n={len(state_data)})"
                    plt.title(title)
                    
                    # Annotating with R-squared and regression equation
                    plt.annotate(label, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top', backgroundcolor='white')

                    
                    plot_filename = os.path.join(output_folder, f"{title.replace(' ', '_').replace('.', '')}.png")
                    if not os.path.exists(output_folder):  # Check if the directory exists
                        os.makedirs(output_folder)  # Create the directory if it does not exist
                    try:
                        plt.savefig(plot_filename)
                        print(f"Plot saved to {plot_filename}")
                    except Exception as e:
                        print(f"Failed to save plot: {plot_filename}")
                        traceback.print_exc()
                    # Show plot
#                     plt.show()
                    plt.close()
                    
                    
                    # Add results to the list
                    results.append({
                        'State': state,
                        'Variable': config['x_col'],
                        'R_squared': r_squared,
                        'N_Observations': len(state_data)
                    })
                else:
                    print(f"Regression could not be performed for {state} due to insufficient data or perfect collinearity.")
            else:
                print(f"Not enough data to plot for {state}.")

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Function to convert date strings to date objects
def convert_date(date_str):
    try:
        return datetime.strptime(date_str, "%d%m%Y").strftime('%Y-%m-%d')
    except ValueError:
        # If there is a ValueError, return a placeholder or indication of the error
        return "Invalid date"
    
# Function to convert date strings to datetime objects
def convert_to_datetime(date_str):
    try:
        return datetime.strptime(date_str, "%d%m%Y")
    except ValueError:
        return pd.NaT  # Use Not a Time (NaT) for invalid or missing dates
    
def plot_and_collect_regression_results(dfs, plots_config, output_folder):
    results = []
    for df in dfs:  # Loop through each DataFrame in the list
        # Extract DataFrame name from the second column, taking the first non-null value and splitting it
        if df.columns.size > 1:
            second_col_name = df.columns[1]
            df_name = second_col_name.split('_')[1]
        else:
            df_name = 'Unknown DataFrame'
        for config in plots_config:  # Access each plot configuration
            # Extract configuration details
            x_col = f"{config['x_col']}_{df_name}"
#             x_col = config['x_col']
            y_col = config['y_col']
            color = config['color']
            xlabel = config['xlabel']
            ylabel = config['ylabel']
            state_col = config['state']  # The column name for state, assumed to be specified in the config

            if state_col in df.columns:
                grouped = df.groupby(state_col)  # Group the DataFrame by the state column
                for state, group in grouped:
                    # Drop rows with NaN in x_col or y_col to ensure accurate plotting and analysis
                    clean_group = group.dropna(subset=[x_col, y_col])
                    # Check if X or Y is empty or all NaNs
                    if clean_group[x_col].dropna().empty or clean_group[y_col].dropna().empty:
                        print(f"⚠️ Skipping due to empty or NaN-only data in x_col: {x_col} or y_col: {y_col}")
                        continue
                    num_observations = len(clean_group)  # Number of observations

                    title = f"{df_name} - {state} - {config['title']} - N={num_observations}"

                    # Perform regression analysis on the group
                    X = sm.add_constant(clean_group[x_col], has_constant='add')  # Add a constant for the intercept
                    model = sm.OLS(clean_group[y_col], X, missing='drop').fit()  # Fit the model

                    # Create scatter plot with regression line and formula
                    plt.figure(figsize=(10, 6))
                    plt.scatter(clean_group[x_col], clean_group[y_col], color=color, alpha=0.5)
                    plt.plot(clean_group[x_col], model.predict(X), color='red')  # Plot the regression line
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    
                    label = f"Formula: {model.params[1]:.2f}x + {model.params[0]:.2f}, R²={model.rsquared:.3f}"
                    plt.annotate(label, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top', backgroundcolor='white')

                    plt.title(title)
                    plt.grid(True)
                    
                    plot_filename = os.path.join(output_folder, f"{title.replace(' ', '_').replace('.', '')}.png")
                    if not os.path.exists(output_folder):  # Check if the directory exists
                        os.makedirs(output_folder)  # Create the directory if it does not exist
                    try:
                        plt.savefig(plot_filename)
                        print(f"Plot saved to {plot_filename}")
                    except Exception as e:
                        print(f"Failed to save plot: {plot_filename}")
                        traceback.print_exc()
                    # Show plot
#                     plt.show()
                    plt.close()

                    # Append results for summary
                    results.append({
                        'DataFrame': df_name,
                        'State': state,
                        'X_column': x_col,
                        'R_squared': model.rsquared,
                        'N_Observations': num_observations
                    })
            else:
                print(f"The state column '{state_col}' does not exist in DataFrame '{df_name}'.")

    # Convert results to a DataFrame
    results_df1 = pd.DataFrame(results)
    return results_df1


# ## sort dates columns

# In[3]:


# Load shapefile
all_mean_data = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_indices_data.csv")
print(all_mean_data)

# ############## put columns date by right order

first_col = all_mean_data.columns[0]
other_cols = all_mean_data.columns[1:]

# Sort the rest by date (first 8 characters) and then by the second part (after "_")
sorted_other_cols = sorted(other_cols, key=lambda x: (x[:8], x[9:]))

# Build the final list of columns: first column + sorted others
final_columns = [first_col] + sorted_other_cols
# Reorder the DataFrame
all_mean_data = all_mean_data[final_columns]
all_mean_data


# ## add yield to all indices raw mean data

# In[4]:


# Load shapefile
gdf = gpd.read_file(r"F:\globalag\clus_sandwich_test\globalAg_clus_with_all_data_yield_normalized_singleCluUnit.gpkg")

def safe_eval_list(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []  # Or np.nan, or [0, 0, 0] depending on what you prefer
    return val  # If it's already a list

gdf['AnnualProduction_lbsAcres'] = gdf['AnnualProduction_lbsAcres'].apply(safe_eval_list)
gdf['ReinsuranceYears'] = gdf['ReinsuranceYears'].apply(safe_eval_list)

# Expand into {year: yield}
expanded_rows = gdf.apply(lambda row: dict(zip(row['ReinsuranceYears'], row['AnnualProduction_lbsAcres'])), axis=1)

# Convert to DataFrame
expanded_df = pd.DataFrame(list(expanded_rows))

# Rename columns
expanded_df.columns = [f"{col}_yield" for col in expanded_df.columns]

# Sort columns
expanded_df = expanded_df.reindex(sorted(expanded_df.columns), axis=1)

# Combine PWId and geometry
final_gdf = pd.concat([gdf[['PWId']], expanded_df], axis=1)
final_gdf


# In[5]:


# Select only PWId and yield columns from the expanded GeoDataFrame
yield_cols = [col for col in final_gdf.columns if col.endswith('_yield')]
yield_df = final_gdf[['PWId'] + yield_cols].copy()
yield_df['PWId'] = yield_df['PWId'].str.replace('pa_unt_', '', regex=False)

all_mean_data['PWId'] = all_mean_data['PWId'].astype(int)
yield_df['PWId'] = yield_df['PWId'].astype(int)

merged_df = all_mean_data.merge(yield_df, on='PWId', how='left')
# Merge with other_df on PWId
merged_df = all_mean_data.merge(yield_df, on='PWId', how='left')  # Use 'left', 'inner', or 'outer' as needed
merged_df


# ## add missing dates with nans (for later interpolation) and divide to index and years df

# In[6]:


# 1. Extract all existing headers (dates and indices)
index_columns = [col for col in merged_df.columns if '_' in col and not col.endswith('yield')]

# Parse existing dates and indices
existing_dates = set()
existing_indices = set()

for col in index_columns:
    parts = col.split('_')
    if len(parts) >= 2:
        existing_dates.add(parts[0])  # Date part
        existing_indices.add(parts[1])  # Index part

# 2. Get all unique years
years = sorted(set([date[:4] for date in existing_dates]))

# 3. Generate full list of expected headers
expected_headers = []

for year in years:
    start_date = datetime.strptime(f"{year}0701", "%Y%m%d")
    end_date = datetime.strptime(f"{year}1229", "%Y%m%d")
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        for idx in existing_indices:
            expected_headers.append(f"{date_str}_{idx}")
        current_date += timedelta(days=1)

# 4. Identify missing headers
missing_headers = [col for col in expected_headers if col not in merged_df.columns]

# 5. Create and concat missing columns (NaNs)
if missing_headers:
    missing_df = pd.DataFrame(np.nan, index=merged_df.index, columns=missing_headers)
    merged_df = pd.concat([merged_df, missing_df], axis=1)
    merged_df[missing_headers] = merged_df[missing_headers].astype(float)

# 6. Sort columns (PWId, date-index, yields)
date_index_cols = sorted([col for col in merged_df.columns if '_' in col and not col.endswith('yield')])
yield_cols = [col for col in merged_df.columns if col.endswith('yield')]
merged_df = merged_df[['PWId'] + date_index_cols + yield_cols]
# merged_df.to_csv(r"F:\globalag\clus_sandwich_test\globalag_all_indices_data_dates_right_order.csv")

# 7. Create separate DataFrames for each index and year

index_year_dfs = dict()

# Loop over each index
for idx in existing_indices:
    # Select all columns ending with this index
    idx_cols = [col for col in date_index_cols if col.endswith(f"_{idx}")]
    
    # Group idx_cols by year
    cols_by_year = defaultdict(list)
    for col in idx_cols:
        year = col.split('_')[0][:4]  # extract year from date
        cols_by_year[year].append(col)
    
    # For each year, build the DataFrame
    for year, cols in cols_by_year.items():
        yield_col = f"{year}_yield"
        selected_cols = ['PWId'] + cols
        if yield_col in merged_df.columns:
            selected_cols.append(yield_col)
        key = f"{idx}_{year}"
        index_year_dfs[key] = merged_df[selected_cols].copy()

# ✅ Summary
print(f"Generated {len(index_year_dfs)} DataFrames: {list(index_year_dfs.keys())}")


# # smooth data - interpolation to final df/csv

# In[7]:


def smooth_row_data(row, window_length=5, polyorder=2):
    try:
        return pd.Series(
            savgol_filter(row.values, window_length=min(window_length, len(row)), polyorder=min(polyorder, window_length - 1)),
            index=row.index
        )
    except ValueError:
        return row

# Process each DataFrame
for key, df in index_year_dfs.items():
    # Identify index/date columns (not PWId or yield)
    data_cols = [col for col in df.columns if not col.startswith('PWId') and not col.endswith('_yield')]

    # Interpolate and fill missing values
    df[data_cols] = df[data_cols].interpolate(axis=1)
    df[data_cols] = df[data_cols].ffill(axis=1).bfill(axis=1)

    # Apply smoothing
    df[data_cols] = df[data_cols].apply(smooth_row_data, axis=1)

    # Update the dictionary
    index_year_dfs[key] = df


# In[8]:


index_year_dfs['NDVI_2019']


# ## add new indices (max, AUC and more)

# In[9]:


import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Fast vectorized metrics calculation ---
def calculate_basic_metrics_fast(df):
    # Detect data columns (exclude PWId and yield)
    data_cols = [
        col for col in df.columns
        if (isinstance(col, str) and not col.startswith('PWId') and not col.endswith('_yield'))
        or isinstance(col, pd.Timestamp)
    ]
    
    # Convert only string date columns to datetime safely
    if any(isinstance(col, str) for col in data_cols):
        new_columns = {}
        for col in data_cols:
            if isinstance(col, str):
                date_part = col.split('_')[0]
                if len(date_part) == 8 and date_part.isdigit():
                    try:
                        new_columns[col] = pd.to_datetime(date_part, format="%Y%m%d")
                    except Exception:
                        new_columns[col] = col
                else:
                    new_columns[col] = col
        df.rename(columns=new_columns, inplace=True)
        data_cols = [col for col in df.columns if isinstance(col, pd.Timestamp)]

    # Core calculations vectorized
    max_value = df[data_cols].max(axis=1)
    max_date = df[data_cols].idxmax(axis=1)
    max_doy = max_date.apply(lambda x: x.dayofyear if pd.notnull(x) else np.nan)
    auc_total = np.trapz(df[data_cols].values, dx=1, axis=1)

    monthly_max = {}
    monthly_auc = {}
    for month in [7, 8, 9, 10, 11]:  # July to November
        month_cols = [col for col in data_cols if col.month == month]
        if month_cols:
            monthly_max[month] = df[month_cols].max(axis=1)
            monthly_auc[month] = np.trapz(df[month_cols].values, dx=1, axis=1)
        else:
            monthly_max[month] = np.nan
            monthly_auc[month] = np.nan

    # Build result DataFrame
    metrics_df = pd.DataFrame({
        'PWId': df['PWId'],
        'max_value': max_value,
        'max_date': max_date,
        'max_doy': max_doy,
        'AUC_total': auc_total,
        'max_July': monthly_max[7],
        'max_August': monthly_max[8],
        'max_September': monthly_max[9],
        'max_October': monthly_max[10],
        'max_November': monthly_max[11],
        'AUC_July': monthly_auc[7],
        'AUC_August': monthly_auc[8],
        'AUC_September': monthly_auc[9],
        'AUC_October': monthly_auc[10],
        'AUC_November': monthly_auc[11],
    })

    return metrics_df

# --- Parallel processing function ---
def process_index_df(key_df_pair):
    key, df = key_df_pair
    print(f"🔵 Processing {key}")
    
    metrics_df = calculate_basic_metrics_fast(df)
    updated_df = df.copy()
    metrics_by_id = metrics_df.set_index('PWId')

    for metric_col in metrics_df.columns:
        if metric_col == 'PWId':
            continue
        new_col_name = f"{key}_{metric_col}"
        updated_df[new_col_name] = metrics_by_id[metric_col].reindex(updated_df['PWId']).values

    # Rename datetime columns back to "YYYYMMDD_index" format
    date_cols = [col for col in updated_df.columns if isinstance(col, pd.Timestamp)]
    rename_dict = {}
    index_name = key.split('_')[0]  # Example: "NDVI" from "NDVI_2020"
    for col in date_cols:
        new_name = f"{col.strftime('%Y%m%d')}_{index_name}"
        rename_dict[col] = new_name

    updated_df.rename(columns=rename_dict, inplace=True)

    print(f"✅ Done {key}")
    return key, updated_df

# --- Run everything in parallel ---
index_years_new_df = {}

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_index_df, item) for item in index_year_dfs.items()]
    for future in as_completed(futures):
        key, updated_df = future.result()
        index_years_new_df[key] = updated_df


# In[10]:


index_years_new_df['NDMI_2019']


# ## merge all indices df to one df

# In[13]:


# Step 1: Set PWId as index in each DataFrame
indexed_dfs = [df.set_index('PWId') for df in index_years_new_df.values()]

# Step 2: Drop duplicate columns before joining
# Keep track of already used columns
used_columns = set()

cleaned_dfs = []
for df in indexed_dfs:
    # Keep only new columns
    new_cols = [col for col in df.columns if col not in used_columns]
    cleaned_dfs.append(df[new_cols])
    used_columns.update(new_cols)

# Step 3: Merge all cleaned DataFrames
final_all_indices_df = reduce(lambda left, right: left.join(right, how='outer'), cleaned_dfs)

# Step 4: Reset index
final_all_indices_df = final_all_indices_df.reset_index()

print("✅ Merged horizontally by PWId with no duplicates!")
print(final_all_indices_df.shape)
print(final_all_indices_df.head())





# In[23]:


# final_all_indices_df.to_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices.csv")


# ## add CLUID and irr data

# In[14]:


gdf2 = gpd.read_file(r"F:\globalag\clus_sandwich_test\globalAg_clus_with_all_data.gpkg")
gdf2.info()
columns_to_keep = ['PWId', 'CLUID', 'PracticeCode']
new_gdf2 = gdf2[columns_to_keep]

final_all_indices_df = final_all_indices_df.loc[:, ~final_all_indices_df.columns.duplicated()]

final_all_indices_df_wiht_irr = final_all_indices_df.merge(new_gdf2, on='PWId', how='left')
final_all_indices_df_wiht_irr.to_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_v2_singleCluUnit.csv")
final_all_indices_df_wiht_irr


# # plots

# ## indices plots by irr group vs its year yield

# In[16]:


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

final_all_indices_df_wiht_irr = final_all_indices_df_wiht_irr[~final_all_indices_df_wiht_irr[yield_cols].isna().all(axis=1)]




# Output directory
plot_dir = r"F:\globalag\clus_sandwich_test\globalag_indices_by_irr_plots_v2_singleCluUnit"
os.makedirs(plot_dir, exist_ok=True)

# Yield columns and fast lookup per year
yield_columns = [col for col in final_all_indices_df_wiht_irr.columns if 'yield' in col.lower()]
yield_by_year = {col.split('_')[0]: col for col in yield_columns if '_' in col}

# Metric keywords
metric_keywords = {
    'max_value', 'AUC_total', 'max_July', 'max_August', 'max_September',
    'max_October', 'max_November', 'AUC_July', 'AUC_August', 'AUC_September',
    'AUC_October', 'AUC_November'
}

r2_results = []

# Main loop
for irr_group_value, group_df in final_all_indices_df_wiht_irr.groupby('PracticeCode'):
    print(f"🔵 PracticeCode: {irr_group_value}")

    # Filter only valid index-metric columns
    metric_cols = [
        col for col in group_df.columns 
        if '_' in col and any(col.endswith(metric) for metric in metric_keywords)
    ]

    for col in metric_cols:
        parts = col.split('_')
        if len(parts) < 3:
            continue

        index, year, metric = parts[0], parts[1], '_'.join(parts[2:])
        if metric not in metric_keywords or year not in yield_by_year:
            continue

        yield_col = yield_by_year[year]
        x = group_df[col]
        y = group_df[yield_col]

        # Mask invalid data early
        valid = x.notna() & y.notna() & (y > 0)
        if valid.sum() < 2:
            continue

        x = x[valid].values.reshape(-1, 1)
        y = y[valid].values

        # Fit model and compute R²
        model = LinearRegression().fit(x, y)
        r2 = r2_score(y, model.predict(x))
        r2 = round(r2, 2)
        r2_results.append({
            'irr_group': irr_group_value,
            'index': index,
            'year': year,
            'metric': metric,
            'yield_column': yield_col,
            'r2_score': r2
        })

        # Plot and save
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, alpha=0.7, label='Data points')
        plt.plot(x, model.predict(x), color='red', label=f'Trendline\n$R^2$={r2:.2f}')
        plt.xlabel(f"{index} {metric} ({year})")
        plt.ylabel(yield_col)
        plt.title(f"Irrigation Group: {irr_group_value}\n{index} {metric} vs {yield_col} ({year})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"{irr_group_value}_{index}_{year}_{metric}.png".replace('/', '-')
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

# Final R² DataFrame
r2_df = pd.DataFrame(r2_results)
r2_df_sorted = r2_df.sort_values(by='r2_score', ascending=False)

print("✅ All R² scores collected and sorted!")
r2_df_sorted
r2_df_sorted.to_csv(r"F:\globalag\clus_sandwich_test\globalag_indices_by_irr_vs_yield_r2_v2_singleCluUnit.csv")


# In[17]:


final_all_indices_df_wiht_irr


# In[18]:


r2_df_sorted


# # new data csv by thresholds

# In[19]:


# Load your files
df_main = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_v2_singleCluUnit.csv")       # File with many columns including PWId
df_acres = gpd.read_file("F:\globalag\clus_sandwich_test\globalAg_clus_with_all_data.gpkg")     # File with PWId and acres
print(f"Rows: {df_main.shape[0]}, Columns: {df_main.shape[1]}")

print(df_acres.info())
# Step 1: Get list of PWId with acres > 5
valid_pwid = df_acres[df_acres['PlantedAcres'] > 5]['PWId'].unique()

# Step 2: Filter the first file using that list
filtered_df = df_main[df_main['PWId'].isin(valid_pwid)]
print(filtered_df.head())
print(f"Rows: {filtered_df.shape[0]}, Columns: {filtered_df.shape[1]}")






# In[20]:


# Identify yield columns
yield_cols = [col for col in filtered_df.columns if 'yield' in col.lower()]

# Apply the filter to each yield column
for col in yield_cols:
    filtered_df[col] = filtered_df[col].where((filtered_df[col] >= 100) & (filtered_df[col] <= 1500), pd.NA)


# In[23]:


filtered_df.to_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit.csv")


# In[ ]:




