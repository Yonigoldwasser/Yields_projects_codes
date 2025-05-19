#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
from scipy.signal import savgol_filter
import os
import traceback


# # plot functions

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


# # Imports

# In[3]:


rvi = pd.read_parquet(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\rvi_group1only1perUnit_geeZonalStats_mean.parquet")
# rvi.to_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\rvi_group1only1perUnit_geeZonalStats_mean.csv")


# In[4]:


lai = pd.read_parquet(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_zonal_lai_stats_mean_fixed.parquet")
lai.head()


# In[5]:


prism = pd.read_parquet(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\PRISM_proagYields212223_MarchOct2023.parquet")
print(prism.info())
print(prism.head())


# In[6]:


PA2023 = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\proag_22_23_24_2023_yiled_data_1clu_1unit_no_dup.gpkg")
print(PA2023.head())


# In[7]:


PWId_keys = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\IL_23\PWId_commonLand_keys.csv")
PWId_keys_map = dict(zip(PWId_keys['commonland'], PWId_keys['PWId']))


# # PD 2023

# In[8]:


elavation_slope = pd.read_parquet(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\elevations_slopes.parquet")


# In[9]:


PD_2023 = pd.read_parquet(r"C:\Users\User\Documents\proag_22_23_24_with_yields_processed23proagHistoryJoined.parquet")
# print('PD_2023', PD_2023.info())
PD_2023 = PD_2023[PD_2023['crop_code'] == 41]
# print('PD_2023', PD_2023.info())
columns_to_keep = ['CLUID', 'proag23_papd_0']  # Your list of desired column names
PD_2023_corn_edited = PD_2023[[col for col in columns_to_keep if col in PD_2023.columns]]

PD_2023_corn_edited = PD_2023_corn_edited.replace('', pd.NA)  # Treat empty strings as NA
PD_2023_corn_edited = PD_2023_corn_edited.dropna()


print('PD_2023_corn_edited',PD_2023_corn_edited.info())
print(PD_2023_corn_edited.head())


# In[10]:


PD_2023_corn_edited['PWId'] = PD_2023_corn_edited['CLUID'].map(PWId_keys_map)
PD_2023_corn_edited = PD_2023_corn_edited.drop(columns=['CLUID'])
print(len(PD_2023_corn_edited))
PD_2023_corn_edited.head()


# In[11]:


# Replace 'ID' and 'date_column' with your actual column names
PWId_PDreal = {
    row['PWId']: datetime.strptime(row['proag23_papd_0'], "%Y-%m-%d")
    for _, row in PD_2023_corn_edited.iterrows()
}

for i, (k, v) in enumerate(PWId_PDreal.items()):
    if i >= 5:
        break
    print(f"{k}: {v}")



# # PA 2023 data Corn

# In[12]:


PA2023 = PA2023[PA2023['crop_name'] == 'CORN']
PA2023['PWId'] = PA2023['commonland'].map(PWId_keys_map)
PA2023_yield_map = dict(zip(PA2023['PWId'], PA2023['unit_number_yield']))
PA2023_state_map = dict(zip(PA2023['PWId'], PA2023['state']))
PA2023_clucalcula_map = dict(zip(PA2023['PWId'], PA2023['clucalcula']))

PA2023.info()
# print('first', PA2023.head())
PA2023_subset = PA2023[['PWId', 'unit_number_yield', 'state', 'clucalcula', 'commonland']]
PA2023_subset.dropna(subset=['PWId'], inplace=True)
PA2023_subset['PWId'] = PA2023_subset['PWId'].astype(int)
PA2023_subset.set_index('PWId', inplace=True)
PA2023_subset = PA2023_subset.sort_index()

print('second', PA2023_subset.head())


# In[ ]:





# # each index df

# In[13]:


group1_data = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_data_mean_sorted_csv.csv")
group1_data.info()


# In[14]:


group1_data_mean = group1_data.groupby('PWId').mean()
# print(group1_data_mean.info())
print(group1_data_mean.head())


# In[15]:


# edit PD dict to have only clus that are in group1 data
PD_filtered_dict = {k: v for k, v in PWId_PDreal.items() if k in group1_data_mean.index}
print(len(group1_data_mean))
len(PD_filtered_dict)
PD_filtered_dict_df = pd.DataFrame(list(PD_filtered_dict.items()), columns=['Key', 'Value'])
# PD_filtered_dict_df.to_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\PD_filtered_dict_df_v4.csv", index=False)


# In[16]:


# Extract unique indices - dataframe for each index
indices = set(col.split('_')[1] for col in group1_data_mean.columns)
indices_list = []
indices_dict = {}

for index in indices:
    # Filter columns for each specific index
    columns = [col for col in group1_data_mean.columns if col.endswith(f'_{index}')]
    df = group1_data_mean[columns]
    df.name = f"df_{index}"  # Set a name attribute
    indices_list.append(df)
    indices_dict[f"df_{index}"] = group1_data_mean[columns]


# Now printing each DataFrame's name
for df in indices_list:
    print(df.name)


# In[17]:


for df in indices_list:
    print(df.head())


# In[18]:


def smooth_row_data(row, window_length=5, polyorder=2):
    # This example just returns the row for simplicity; replace with your smoothing logic
    return row

# Modify each DataFrame in the list
for index in range(len(indices_list)):
    indices_list[index] = indices_list[index].interpolate(axis=1)  # Interpolate in-place
    indices_list[index] = indices_list[index].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)  # Fill remaining NaNs forward first, then backward
    indices_list[index] = indices_list[index].apply(smooth_row_data, axis=1)  # Apply smoothing




# In[19]:


for i in indices_list:
    i = i.interpolate(axis=1).ffill(axis=1).bfill(axis=1)
    i = i.apply(smooth_row_data, axis=1)



# In[20]:


indices_list1 = indices_list


# In[21]:


for df in indices_list1:
    print(df.head())


# In[22]:


import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from datetime import datetime

def smooth_row_data(row, window_length=5, polyorder=2):
    return pd.Series(savgol_filter(row, window_length, polyorder), index=row.index)

# def find_first_above_045_below_030(df):
#     date_format = "%d%m%Y"
    
#     # Check if the columns are datetime; if not, convert them
#     if not isinstance(df.columns, pd.DatetimeIndex):
#         rename_dict = {col: col.split('_')[0] for col in df.columns if isinstance(col, str) and '_' in col}
#         df.rename(columns=rename_dict, inplace=True)
#         df.columns = pd.to_datetime(df.columns, format=date_format)
    
#     df_filtered = df.loc[:, df.columns > pd.Timestamp('2023-03-01')]
    
#     first_above_045_dates = {}
#     last_below_030_dates = {}  # Changed to track the last transition below 0.3
    
# #     df_run = df_filtered.iloc[-1:]
#     for index, row in df_filtered.iterrows():
#         # Find the first time above 0.45
#         above_045_date = None
#         for date, value in row.items():
#             if value > 0.40:  # Note: using 0.40 as per original code
#                 above_045_date = date
#                 first_above_045_dates[index] = date
#                 break
                
#         # If no start date (above 0.45), skip this row
#         if index not in first_above_045_dates:
#             continue
        
#         # Now check if there are any instances where value goes above 0.3 after the first_above_045_date
#         dates_after_threshold = row[row.index >= first_above_045_dates[index]]
        
#         # Find periods where values are above 0.3
#         above_030_periods = []
#         current_period_start = None
        
#         for date, value in dates_after_threshold.items():
#             if value > 0.3 and current_period_start is None:
#                 # Start a new period above 0.3
#                 current_period_start = date
#             elif value <= 0.3 and current_period_start is not None:
#                 # End the current period and record it
#                 above_030_periods.append((current_period_start, date))
#                 current_period_start = None
        
#         # If we're still in a period above 0.3 at the end of the dataset
#         if current_period_start is not None:
#             above_030_periods.append((current_period_start, dates_after_threshold.index[-1]))
        
#         # If we found any periods above 0.3
#         if above_030_periods:
#             # If the last period ends at the last date in the dataset, it means 
#             # the values never dropped below 0.3 after the last rise above 0.3
#             last_period_end = above_030_periods[-1][1]
#             if last_period_end == dates_after_threshold.index[-1]:
#                 last_below_030_dates[index] = last_period_end
#             else:
#                 # Otherwise, use the date when values last dropped below 0.3
#                 last_below_030_dates[index] = last_period_end
#         else:
#             # If no periods above 0.3 were found after the first_above_045_date
#             last_below_030_dates[index] = dates_after_threshold.index[-1]
    
#     return first_above_045_dates, last_below_030_dates

# method from value max date backward and onwards to 0.4 and 0.4
def find_first_above_045_below_030(df, PD_filtered_dict):
    date_format = "%d%m%Y"
    first_above_045_dates = PD_filtered_dict
    if not isinstance(df.columns, pd.DatetimeIndex):
        rename_dict = {col: col.split('_')[0] for col in df.columns if isinstance(col, str) and '_' in col}
        df.rename(columns=rename_dict, inplace=True)
        df.columns = pd.to_datetime(df.columns, format=date_format)

    df_filtered = df.loc[:, df.columns > pd.Timestamp('2023-03-01')]

#     first_above_045_dates = {}
    last_below_030_dates = {}

    for index, row in df_filtered.iterrows():
        values = row.values
        dates = row.index

        # Get index of max value
        max_idx = np.argmax(values)
        max_date = dates[max_idx]

#         # Find start date: last date before max where value < 0.4
#         start_date = pd.NaT
#         for i in range(max_idx, -1, -1):
#             if values[i] < 0.4:
#                 start_date = dates[i]
#                 break

        # Find end date: first date after max where value < 0.3
        end_date = pd.NaT
        for i in range(max_idx, len(values)):
            if values[i] < 0.3:
                end_date = dates[i]
                break

        # Only save if both were found and logically ordered
        if not pd.isna(end_date): # and start_date < end_date:
#             first_above_045_dates[index] = start_date
            last_below_030_dates[index] = end_date
    
    # make sure both dicts have the same keys
    common_keys = first_above_045_dates.keys() & last_below_030_dates.keys()
#     print(f"Common keys: {len(common_keys)}")
    # Filter both dicts to keep only common keys
    first_above_045_dates = {k: first_above_045_dates[k] for k in common_keys}
    last_below_030_dates = {k: last_below_030_dates[k] for k in common_keys}
    
    return first_above_045_dates, last_below_030_dates

def calculate_metrics_from_date(df, first_above_045_dates, last_below_030_dates):
    metrics = {
        'PWId': {},
        'max_index': {},
        'max_index_date': {},
        'max_index_DOY': {},
        'AUC': {},
        'AUC_JUN_AUG': {},
        'AUC_above_045': {},
        'AUC_below_030': {},
        'growing_days': {}
    }
    
    for index, row in df.iterrows():
#         print('index', index)
#         print(first_above_045_dates)
        if index in first_above_045_dates and index in last_below_030_dates:
            start_date = first_above_045_dates[index]
            end_date = last_below_030_dates[index]
#             print('start_date', start_date)
#             print('end_date', end_date)
            if start_date < end_date:
                relevant_data = row.loc[start_date:end_date]
#                 print('relevant_data', relevant_data)
                metrics['PWId'][index] = relevant_data.name  # Correct assignment to a dictionary
                metrics['max_index'][index] = relevant_data.max()
                metrics['max_index_date'][index] = relevant_data.idxmax()
                metrics['max_index_DOY'][index] = relevant_data.idxmax().dayofyear
                metrics['AUC'][index] = np.trapz(relevant_data, dx=1)
                
                # Filter data for June-August
                jun_aug_data = relevant_data.loc[(relevant_data.index.month >= 6) & (relevant_data.index.month <= 8)]
                metrics['AUC_JUN_AUG'][index] = np.trapz(jun_aug_data, dx=1)
                # Duration in days
                metrics['growing_days'][index] = (end_date - start_date).days
                # Uncomment these lines if you need to compute these metrics
                # metrics['AUC_above_045'][index] = np.trapz(relevant_data[relevant_data > 0.45], dx=1)
                # metrics['AUC_below_030'][index] = np.trapz(relevant_data[relevant_data < 0.3], dx=1)
                
            else:
                # Invalid period
                for key in metrics:
                    metrics[key][index] = np.nan
        else:
            for key in metrics:
                metrics[key][index] = np.nan
    return metrics


# applying first and last dates for each clu TS by the NDVI df
first_above_045_dates, last_below_030_dates = find_first_above_045_below_030(indices_list1[-1], PD_filtered_dict)

g = pd.DataFrame({
    'Start Date': pd.Series(first_above_045_dates),
    'End Date': pd.Series(last_below_030_dates)
})
# print(g)
# print(len(first_above_045_dates))
# print(len(last_below_030_dates))

# Assuming indices_list1 is the correct list containing your DataFrames
for i, df in enumerate(indices_list1):
    if isinstance(df, pd.DataFrame):
        print(f"\n🔍 Processing DataFrame index {i}")
        print(f"🟡 Original columns:\n{df.columns.tolist()}")

        # Detect if columns are datetime
        columns_are_datetime = isinstance(df.columns, pd.DatetimeIndex)

        # Determine df_name
        df_name = None
        if not columns_are_datetime:
            for col in df.columns:
                if isinstance(col, str) and "_" in col:
                    df_name = col.split("_")[1]
                    print(f"✅ Extracted df_name: {df_name}")
                    break
        else:
            df_name = "NDVI"
            print(f"⚠️ Columns already datetime. Using fallback df_name: {df_name}")

        # If columns are not datetime, validate and clean them
        if not columns_are_datetime:
            valid_cols = []
            for col in df.columns:
                if isinstance(col, str):
                    try:
                        date_str = col.split("_")[0]
                        datetime.strptime(date_str, "%d%m%Y")
                        valid_cols.append(col)
                    except Exception:
                        print(f"⛔ Skipping non-date column: {col}")
            df = df[valid_cols]
            print(f"✅ Valid date columns:\n{df.columns.tolist()}")
        else:
            print(f"✅ Columns already in datetime format.")

        # Convert values to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                print(f"⚠️ Could not convert column {col} to numeric")

        numeric_df = df.select_dtypes(include=[np.number])

        # Only reassign datetime if it's not already done
        if not columns_are_datetime:
            date_strings = [col.split("_")[0] for col in numeric_df.columns]
            column_dates = pd.to_datetime(date_strings, format="%d%m%Y")
            numeric_df.columns = column_dates

        filled_numeric_df = numeric_df.ffill(axis=1).bfill(axis=1)

        if filled_numeric_df.empty:
            print(f"❌ Skipping index {i} — filled_numeric_df is empty")
            continue

        # Calculate metrics safely
        metrics = calculate_metrics_from_date(filled_numeric_df, first_above_045_dates, last_below_030_dates)
        # Assemble metrics DataFrame
        metrics_df = pd.DataFrame({
            f"PWId": df.index.map(metrics['PWId']),
            f"max_index_{df_name}": df.index.map(metrics['max_index']),
            f"max_index_date_{df_name}": df.index.map(metrics['max_index_date']),
            f"max_index_DOY_{df_name}": df.index.map(metrics['max_index_DOY']),
            f"AUC_{df_name}": df.index.map(metrics['AUC']),
            f"AUC_JUN_AUG_{df_name}": df.index.map(metrics['AUC_JUN_AUG']),
            f"AUC_above_045_{df_name}": df.index.map(metrics['AUC_above_045']),
            f"AUC_below_030_{df_name}": df.index.map(metrics['AUC_below_030']),
            f"growing_days_{df_name}": df.index.map(metrics['growing_days'])
        })
        
        
        metrics_df = metrics_df.dropna(subset=['PWId'])
        print(f"🧮 Metrics DF sample:\n{metrics_df.head()}")
        print(len(metrics_df))

        merged_df = pd.merge(df, metrics_df, left_index=True, right_on='PWId', how='left')
        merged_df.set_index('PWId', inplace=True)

        second_merged_df = merged_df.merge(PA2023_subset, left_index=True, right_index=True)
        second_merged_df = second_merged_df.rename(columns={'unit_number_yield': '2023_yield'})
        indices_list1[i] = second_merged_df



# In[23]:


Corn_yield_list = []
for df in indices_list1:
    print(df)
    yield_df = df[df['2023_yield']>0]
    Corn_yield_list.append(yield_df)


# In[24]:


from datetime import datetime

Corn_yield_list_edited = []

for df in Corn_yield_list:
    new_df = df.copy()
    new_columns = {}
    
    for col in new_df.columns:
        if isinstance(col, (datetime, pd.Timestamp)):
            new_name = col.strftime('%d%m%Y') + '_NDVI'
            new_columns[col] = new_name
        elif isinstance(col, str):
            try:
                dt = pd.to_datetime(col)
                new_name = dt.strftime('%d%m%Y') + '_NDVI'
                new_columns[col] = new_name
            except (ValueError, TypeError):
                continue  # Not a datetime string, skip

    new_df.rename(columns=new_columns, inplace=True)
    Corn_yield_list_edited.append(new_df)


# In[25]:


for df in indices_list1:
    print(Corn_yield_list_edited)


# # PRISM data

# In[26]:


print('prism', prism)
prism['date'] = pd.to_datetime(prism['date'])

# Calculate mean temperature
prism['MeanTemp'] = (prism['tmaxMean'] + prism['tminMean']) / 2

# Calculate GDD
prism['GDD'] = prism['MeanTemp'] - 10
prism['GDD'] = prism['GDD'].clip(lower=0)


# adj GDD
# Adjust Tmax and Tmin according to the conditions
prism['Adj_Tmax'] = prism['tmaxMean'].clip(lower=10, upper=30)
prism['Adj_Tmin'] = prism['tminMean'].clip(lower=10)

# Calculate mean temperature using adjusted values
prism['Adj_MeanTemp'] = (prism['Adj_Tmax'] + prism['Adj_Tmin']) / 2

# Calculate GDD
prism['Adj_GDD'] = prism['Adj_MeanTemp'] - 10
# print('prism2', prism)
prism.set_index('PWId', inplace=True)

# add 2023 yield
prism_merge = prism.merge(PA2023_subset, left_index=True, right_index=True)
prism_merge = prism_merge.rename(columns={'unit_number_yield': '2023_yield'})
prism_merge.info()


# In[27]:


prism_merge.head()


# In[28]:


# Convert dictionary dates to datetime if they are not already.
first_above_045_dates = {pwid: pd.to_datetime(date) for pwid, date in first_above_045_dates.items()}
last_below_030_dates = {pwid: pd.to_datetime(date) for pwid, date in last_below_030_dates.items()}

# Create a temporary DataFrame from your date dictionaries
dates_df = pd.DataFrame({
    'Start_Date': first_above_045_dates,
    'End_Date': last_below_030_dates
})
print('dates', dates_df.head())
print('prism_merge', prism_merge.head())

# Merge this dates DataFrame with your original DataFrame to line up start and end dates with each PWId
prism_merge = prism_merge.merge(dates_df, left_index=True, right_index=True, how='left')
print('prism_merge2', prism_merge.head())

# Create a mask to filter rows where the date is within the start and end date range
mask = (prism_merge['date'] >= prism_merge['Start_Date']) & (prism_merge['date'] <= prism_merge['End_Date'])
filtered_prism_merge = prism_merge.loc[mask].drop(columns=['Start_Date', 'End_Date'])

# Show the head of the filtered DataFrame
filtered_prism_merge.head()


# In[29]:


prism_sum = filtered_prism_merge.groupby(filtered_prism_merge.index).agg({
    'pptMean': 'sum',    # Summing up all precipitation mean values
    'GDD': 'sum',        # Summing up all GDD values
    'Adj_GDD': 'sum',    # Summing up all Adjusted GDD values
    '2023_yield': 'first',  # Taking the first yield entry (assuming it doesn't change)
    'state': 'first'     # Taking the first state entry (assuming it's the same for all entries)
})

# Print the first few entries to review the aggregated data
print(prism_sum.head())


# In[30]:


prism_sum


# In[33]:


import pandas as pd

# Make sure PWId is a column (not in the index)
if 'PWId' not in prism_merge.columns:
    prism_merge.reset_index(inplace=True)

# Convert 'date' column to datetime format
prism_merge['date'] = pd.to_datetime(prism_merge['date'])

# Create a formatted date string
prism_merge['date_str'] = prism_merge['date'].dt.strftime('%Y%m%d')

# Set multi-index for pivoting
prism_merge.set_index(['PWId', 'date_str'], inplace=True)

# Define the variable columns to pivot
value_columns = ['pptMean', 'tminMean', 'tmaxMean', 'MeanTemp', 'GDD',
                 'Adj_Tmax', 'Adj_Tmin', 'Adj_MeanTemp', 'Adj_GDD']

# Pivot the table so each date-variable combo becomes a column
pivoted = prism_merge[value_columns].unstack()

# Rename the columns: date_variable
pivoted.columns = [f"{date}_{var}" for var, date in pivoted.columns]

# Bring PWId back as a column
pivoted.reset_index(inplace=True)

# Extract unique metadata columns for each PWId
meta_cols = (
    prism_merge
    .reset_index()
    .drop_duplicates('PWId')[['PWId', '2023_yield', 'state', 'clucalcula', 'commonland']]
)

# Merge the pivoted data with metadata
final_df = pd.merge(meta_cols, pivoted, on='PWId')

# Done! Preview result
print(final_df.head())


# In[36]:


final_df = final_df.set_index('PWId')
final_df


# In[37]:


# Select specific columns from df2 (make sure to include the index)
columns_to_merge = ['pptMean','GDD','Adj_GDD']  # replace with your actual columns
prism_sum_subset = prism_sum[columns_to_merge].copy()
# Merge on index
final_prism_df = final_df.merge(prism_sum_subset, left_index=True, right_index=True, how='left')
final_prism_df


# In[51]:


final_prism_df_for_merge = final_prism_df.drop(columns=['2023_yield', 'state', 'clucalcula', 'commonland'])


# # RVI

# In[38]:


rvi.info()
rvi.head()


# In[40]:


# Convert 'date' column to datetime
rvi['date'] = pd.to_datetime(rvi['date'])
start_date = pd.Timestamp('2023-03-01')
rvi = rvi[rvi['date'] >= start_date]


rvi['DOY'] = rvi['date'].dt.dayofyear

######## average duplicated dates for same PWId - diff footprints ################
# Group by 'PWId' and 'date', and aggregate 'mean' by taking the average
rvi_df = rvi.groupby(['PWId', 'date']).agg({
    'mean': 'mean'  # Averaging the mean values
}).reset_index()
rvi_df.info()
rvi_df.head()


# In[41]:


# If 'Start_Date' and 'End_Date' are already in rvi_df and you intend to replace them, you can drop them:
if 'Start_Date' in rvi_df.columns and 'End_Date' in rvi_df.columns:
    print(rvi_df.columns[1])
    rvi_df = rvi_df.drop(columns=['Start_Date', 'End_Date'])

# print('rvi_df', rvi_df)
# print('dates_df', dates_df)

# Now perform the merge with suffixes to handle any unforeseen overlaps
rvi_df = rvi_df.merge(dates_df, left_on='PWId', right_index=True, how='left')#, suffixes=('', '_dup'))

# Check for duplicated columns after merge and drop if necessary
for col in ['Start_Date', 'End_Date']:
    if col + '_dup' in rvi_df.columns:
        # Assuming the '_dup' columns are the ones from dates_df and you want to keep these
        rvi_df.drop(col, axis=1, inplace=True)
        rvi_df.rename(columns={col + '_dup': col}, inplace=True)

# Filter rows within the specified start and end dates for each PWId
rvi_df = rvi_df[(rvi_df['date'] >= rvi_df['Start_Date']) & (rvi_df['date'] <= rvi_df['End_Date'])]

# Continue with your processing
# print('rvi_dfminusdates', rvi_df)
rvi_df['date_numeric'] = (rvi_df['date'] - rvi_df['Start_Date']).dt.days

# Calculate AUC for each PWId using the trapezoidal rule
auc_results = rvi_df.groupby('PWId').apply(lambda x: np.trapz(x['mean'], x['date_numeric']))
auc_df = auc_results.reset_index(name='AUC')


# Calculate AUC for each PWId using the trapezoidal rule in date range
# Define the date range for the AUC calculation
start_date = pd.to_datetime('2023-06-01')
end_date = pd.to_datetime('2023-08-01')

# Filter the DataFrame to only include rows within the specified date range
rvi_filtered_df = rvi_df[(rvi_df['date'] >= start_date) & (rvi_df['date'] <= end_date)]
# print('rvi_filtered_df', rvi_filtered_df)

# Calculate the AUC for each PWId using the trapezoidal rule only on the filtered data
range_auc_results = rvi_filtered_df.groupby('PWId').apply(lambda x: np.trapz(x['mean'], x['date_numeric']))
# print('range_auc_results', range_auc_results)

# Reset the index to turn the results into a DataFrame
range_auc_df = range_auc_results.reset_index(name='AUC')

rvi_name = 'rvi'
# Find maximum details and other steps
max_details = rvi_df.groupby('PWId').apply(lambda x: x.loc[x['mean'].idxmax()]).reset_index(drop=True)
max_details.rename(columns={'date': f'max_date_{rvi_name}', 'mean': f'max_mean_{rvi_name}'}, inplace=True)
max_details[f'AUC_{rvi_name}'] = auc_df['AUC']
max_details[f'max_DOY_{rvi_name}'] = max_details[f'max_date_{rvi_name}'].dt.dayofyear
max_details[f'AUC_JUN_AUG_{rvi_name}'] = range_auc_df['AUC']
# print(max_details.head())

# Merge with additional data and finalize
max_details = max_details.merge(PA2023_subset, on='PWId', how='left').rename(columns={'max_mean': 'max_rvi', 'unit_number_yield': '2023_yield'})
max_details = max_details.sort_values(by='PWId', ascending=True)
print(max_details)


# In[42]:


rvi_corn = max_details[max_details['PWId'].isin(PA2023_subset.index)]
rvi_corn = rvi_corn[rvi_corn['2023_yield']>0]
# rvi_corn = rvi_corn[rvi_corn['AUC_JUN_AUG']>0]
# rvi_corn = rvi_corn[rvi_corn['max_rvi']<1.5]
rvi_corn.set_index('PWId', inplace=True)

rvi_corn


# # LAI

# In[43]:


lai = lai.groupby('PWId').mean()
lai = lai.interpolate(axis=1)  # Interpolate in-place
print(lai)

#remove rows that are all nans
lai = lai[~lai.isna().all(axis=1)]
lai = lai.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
print('lai2', lai)
lai = lai.apply(smooth_row_data, axis=1)  # Apply smoothing
lai = lai.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

lai = lai.dropna()
lai.info()
lai_df = lai
lai_df


# In[44]:


print(lai_df.index.dtype)
lai_df.index = lai_df.index.map(str)

# Function to extract the date part and sort the columns
def sort_columns_by_date(df):
    sorted_columns = sorted(df.columns, key=lambda x: pd.to_datetime(x.split('_')[0], format='%d%m%Y'))
    return df[sorted_columns]

# Sorting the DataFrame columns
lai_df = sort_columns_by_date(lai_df)

# Display the sorted DataFrame

# lai_df.to_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_lai.csv")
lai_df


# In[45]:


def calculate_metrics_from_date_lai(df, first_above_045_dates, last_below_030_dates):
    metrics = {
        'PWId': {},
        'max_index': {},
        'max_index_date': {},
        'max_index_DOY': {},
        'AUC': {},
        'AUC_JUN_AUG': {},
        'AUC_above_045': {},
        'AUC_below_030': {}
    }
    
    for index, row in df.iterrows():
#         print(index)
#         print(first_above_045_dates)
        if index in first_above_045_dates_new and index in last_below_030_dates:
            start_date = first_above_045_dates_new[index]
            end_date = last_below_030_dates[index]
#             print('start_date', start_date)
#             print('end_date', end_date)
            if start_date < end_date:
                relevant_data = row.loc[start_date:end_date]
#                 print('relevant_data', relevant_data)
                metrics['PWId'][index] = relevant_data.name  # Correct assignment to a dictionary
                metrics['max_index'][index] = relevant_data.max()
                metrics['max_index_date'][index] = relevant_data.idxmax()
                metrics['max_index_DOY'][index] = relevant_data.idxmax().dayofyear
                metrics['AUC'][index] = np.trapz(relevant_data, dx=1)
                
                # Filter data for June-August
                jun_aug_data = relevant_data.loc[(relevant_data.index.month >= 6) & (relevant_data.index.month <= 8)]
                metrics['AUC_JUN_AUG'][index] = np.trapz(jun_aug_data, dx=1)
                
                # Uncomment these lines if you need to compute these metrics
                # metrics['AUC_above_045'][index] = np.trapz(relevant_data[relevant_data > 0.45], dx=1)
                # metrics['AUC_below_030'][index] = np.trapz(relevant_data[relevant_data < 0.3], dx=1)
        else:
            # Set all metrics to NaN if the conditions are not met but keep the index
            metrics['PWId'][index] = index
            metrics['max_index'][index] = np.nan
            metrics['max_index_date'][index] = np.nan
            metrics['max_index_DOY'][index] = np.nan
            metrics['AUC'][index] = np.nan
            metrics['AUC_JUN_AUG'][index] = np.nan
            metrics['AUC_above_045'][index] = np.nan
            metrics['AUC_below_030'][index] = np.nan

    return metrics

df_name = 'lai'

for col in lai_df.columns:
    try:
        lai_df.loc[:, col] = pd.to_numeric(lai_df[col])
    except ValueError:
        continue
numeric_df_lai = lai_df.select_dtypes(include=[np.number])
date_format = "%d%m%Y"
column_dates_lai = pd.to_datetime(numeric_df_lai.columns.str.split('_').str[0], format=date_format)
numeric_df_lai.columns = column_dates_lai

filled_numeric_df_lai = numeric_df_lai.ffill(axis=1).bfill(axis=1)
# print(filled_numeric_df_lai)

# Calculate metrics using the dates obtained from df_NDVI
first_above_045_dates_new = {str(float(key)): value for key, value in first_above_045_dates.items()}
first_below_03_dates_new = {str(float(key)): value for key, value in last_below_030_dates.items()}
# print(first_below_03_dates_new)

metrics_lai = calculate_metrics_from_date_lai(filled_numeric_df_lai, first_above_045_dates_new, first_below_03_dates_new)
# print(list(metrics_lai.items())[:5])

# Combine the metrics into a single DataFrame
metrics_df_lai = pd.DataFrame({
    f"PWId": lai_df.index.map(metrics_lai['PWId']),
    f"max_index_{df_name}": lai_df.index.map(metrics_lai['max_index']),
    f"max_index_date_{df_name}": lai_df.index.map(metrics_lai['max_index_date']),
    f"max_index_DOY_{df_name}": lai_df.index.map(metrics_lai['max_index_DOY']),
    f"AUC_{df_name}": lai_df.index.map(metrics_lai['AUC']),
    f"AUC_JUN_AUG_{df_name}": lai_df.index.map(metrics_lai['AUC_JUN_AUG']),
#     f"AUC_above_045_{df_name}": lai_df.index.map(metrics_lai['AUC_above_045']),
#     f"AUC_below_030_{df_name}": lai_df.index.map(metrics_lai['AUC_below_030'])
})

# print('firstdf',lai_df.head())
# print('metrics df', metrics_df_lai.head())

merged_df_lai = pd.merge(lai_df, metrics_df_lai, left_index=True, right_on='PWId', how='left')
merged_df_lai.set_index('PWId', inplace=True)
merged_df_lai.index = merged_df_lai.index.astype(float).astype(int)

# print('merged_df_lai', merged_df_lai)
# print('PA2023_subset', PA2023_subset.head())

second_merged_df_lai = merged_df_lai.merge(PA2023_subset, left_index=True, right_index=True)
second_merged_df_lai = second_merged_df_lai.rename(columns={'unit_number_yield': '2023_yield'})
print('second_merged_df_lai', second_merged_df_lai)


# In[46]:


final_lai_df = second_merged_df_lai[second_merged_df_lai['2023_yield']>0]
final_lai_df = final_lai_df.dropna()
print(final_lai_df)


# # elevation slope

# In[47]:


elavation_map = dict(zip(elavation_slope['PWId'], elavation_slope['mean_elevation']))
slope_map = dict(zip(elavation_slope['PWId'], elavation_slope['mean_slope']))
elavation_slope.head()


# # all data combined

# In[49]:


print(Corn_yield_list_edited[-1])


# In[52]:


# all data combines (includes new calculate indices and interpolated indices date values)


# Rewritten version with switched variable names and dynamic column inclusion
all_dataframes = Corn_yield_list_edited  # Original: Corn_yield_list
merged_dfs = []
static_cols = None  # Original: first_df_cols

for idx, single_df in enumerate(all_dataframes):
    # Extracting a unique name identifier based on column structure
    unique_tag = single_df.columns[-5].split("_")[2]
    print('unique_tag:', unique_tag)

    # Include all columns except the constant ones used for merging later
    variable_columns = [col for col in single_df.columns if col not in ['2023_yield', 'state', 'clucalcula', 'commonland']]
    
    # Rename these columns to include the tag
    renamed_columns = {col: f"{col}_{unique_tag}" for col in variable_columns}
    dynamic_df = single_df[variable_columns].rename(columns=renamed_columns)

    # Clean list values (if any)
    dynamic_df = dynamic_df.apply(lambda x: x[0] if isinstance(x, list) else x)

    if idx == 0:
        static_cols = single_df[['2023_yield', 'state', 'clucalcula', 'commonland']]
        combined = pd.concat([static_cols, dynamic_df], axis=1)
    else:
        combined = dynamic_df

    merged_dfs.append(combined)

# Concatenate along columns
final_combined_df = pd.concat(merged_dfs, axis=1)

# Merge LAI, RVI, and PRISM using all columns
final_combined_df = final_combined_df.merge(final_lai_df, left_index=True, right_index=True)
final_combined_df = final_combined_df.merge(rvi_corn, left_index=True, right_index=True)
final_combined_df = final_combined_df.merge(final_prism_df_for_merge, left_index=True, right_index=True)

# Add terrain information
final_combined_df['mean_slope'] = final_combined_df.index.map(slope_map)
final_combined_df['mean_elevation'] = final_combined_df.index.map(elavation_map)

# Group yield values
yield_conditions = [
    final_combined_df['2023_yield'] < 100,
    (final_combined_df['2023_yield'] >= 100) & (final_combined_df['2023_yield'] <= 180),
    final_combined_df['2023_yield'] > 180
]
yield_labels = ['Low', 'Medium', 'High']
final_combined_df['yield_group'] = np.select(yield_conditions, yield_labels, default=np.nan)

# Save to CSV
output_path = r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_including_indices_dates_interpolation_v4_20250417_PDreal.csv"
final_combined_df.to_csv(output_path)

# Preview
print(final_combined_df.head())


# In[ ]:




