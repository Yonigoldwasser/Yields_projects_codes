#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw


# # combined

# In[79]:


path = r"C:\Users\User\Documents\Yields_project\Proag_Yields\proAg_yield_pw_units_07012024\years_comparison_PA_NDVI_TS_23_21_19_from_all_history.csv"
df = pd.read_csv(path)

# Ensure the date column is in datetime format
df['Date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df.loc[(df['2019_yield'] != 0) & (df['2021_yield'] != 0) & (df['2023_yield'] != 0)]
df = df.rename(columns={'PWId': 'CLUID'})
df


# In[29]:


df_no_CLUID_dup = df.drop_duplicates(subset='CLUID', keep='first')

# Combine all yields into a single series
all_yields = pd.concat([df_no_CLUID_dup['2019_yield'], df_no_CLUID_dup['2021_yield'], df_no_CLUID_dup['2023_yield']])

# Plot the histogram for all years combined
plt.figure(figsize=(10, 7))
counts, bins, patches = plt.hist(all_yields, bins=30, alpha=0.75, color='blue')
plt.title('Histogram of All Yield Values')
plt.xlabel('Yield')
plt.ylabel('Frequency')

# Add numbers on top of each column in the combined histogram
for count, bin in zip(counts, bins[:-1]):
    # Calculate the center of each bin
    bin_center = bin + (bins[1] - bins[0]) / 2
    # Annotate the count above each patch (column)
    plt.annotate(str(int(count)), xy=(bin_center, count), xycoords='data', 
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

# Show the combined histogram
plt.show()

# Create individual histograms for each yield year
for year in ['2019', '2021', '2023']:
    plt.figure(figsize=(10, 7))
    counts, bins, patches = plt.hist(df_no_CLUID_dup[f'{year}_yield'], bins=30, alpha=0.75, color='blue')
    plt.title(f'Histogram of {year} Yield Values')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    
    # Add numbers on top of each column for individual year histograms
    for count, bin in zip(counts, bins[:-1]):
        bin_center = bin + (bins[1] - bins[0]) / 2
        plt.annotate(str(int(count)), xy=(bin_center, count), xycoords='data', 
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    # Show the histogram for the individual year
    plt.show()


# In[81]:


print(df.info())
# Function to categorize yields
def categorize_yield(yield_value):
    print(yield_value)
    if yield_value < 180:
        return 'Low'
    elif 150 <= yield_value <= 220:
        return 'Medium'
    else:
        return 'High'

# Apply the function to each yield column and create a new group column
for column in df.columns:
    if 'yield' in column:
        print(column)
        df[column + '_group'] = df[column].apply(categorize_yield)

df


# In[111]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.signal import savgol_filter
from IPython.display import display


def smooth_data(group, iterations=3):
    group = group.sort_values(by='Date')
    smoothed_values = group['mean'].copy()
    
    for _ in range(iterations):
        for i in range(1, len(group) - 2):
            if group.iloc[i]['mean'] < group.iloc[i - 1]['mean'] and group.iloc[i]['mean'] < group.iloc[i + 1]['mean']:
                smoothed_values.iloc[i] = (group.iloc[i - 1]['mean'] + group.iloc[i + 1]['mean']) / 2
            
            # Additional check and adjustment
            if group.iloc[i]['mean'] < group.iloc[i - 1]['mean'] and group.iloc[i + 1]['mean'] <= 1 and group.iloc[i + 2]['mean'] > group.iloc[i]['mean']:
                smoothed_values.iloc[i + 1] = (group.iloc[i + 2]['mean'] + group.iloc[i - 1]['mean']) / 2
        group['mean'] = smoothed_values
    group['Smoothed'] = smoothed_values
    return group


def smooth_data_golay(group, window_length=5, polyorder=2):
    group = group.sort_values(by='Date')
    group['Smoothed'] = savgol_filter(group['mean'], window_length=window_length, polyorder=polyorder)
    return group

# Function to normalize dates
def normalize_dates(group):
    group['Normalized Date'] = group['Date'].apply(lambda x: x.replace(year=2000))
    group['Normalized Date'] = pd.to_datetime(group['Normalized Date'])
    return group

# Define distance functions
def euclidean_distance(ts1, ts2):
    return np.sqrt(np.sum((ts1 - ts2)**2))

def dtw_distance(ts1, ts2):
    distance, paths = dtw.warping_paths(np.array(ts1, dtype=np.double), np.array(ts2, dtype=np.double))
    return distance

def spectral_angle_mapper(ts1, ts2):
    dot_product = np.dot(ts1, ts2)
    norm_ts1 = np.linalg.norm(ts1)
    norm_ts2 = np.linalg.norm(ts2)
    return np.arccos(dot_product / (norm_ts1 * norm_ts2))

# Define aggregation methods dynamically
def get_aggregation_methods(df):
    aggregation_methods = {}
    for col in df.columns:
        if col == 'mean':
            aggregation_methods[col] = 'mean'  # Average for 'mean' column
        elif col != 'Date':
            aggregation_methods[col] = 'first'  # 'First' for all other columns
    return aggregation_methods

def interpolate_series(ts, date_index, full_date_range):
    series = pd.Series(ts, index=date_index)
    series = series.reindex(full_date_range).interpolate(method='linear').ffill().bfill()
    return series.to_numpy()

def years_NDVI_TS_comparison(df, YearOne, YearTwo, YearThree):
    # Ensure the date column is in datetime format
    df['Date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.drop('date', axis=1)
    df = df[df['mean'] >= 0.3]
    
    # Initialize a list to store all results
    all_results = []
    interpolated_smoothed_data_all = {}

    # Loop through each CLUID
    for cluid, cluid_group in df.groupby('CLUID'):
        print('cluid', cluid)
        interpolated_smoothed_data_all[CLUID] = {}
        # Generate the aggregation dictionary based on the current group columns
        aggregation_methods = get_aggregation_methods(cluid_group)

        # Group by 'date', applying defined aggregation methods
        cluid_group = cluid_group.groupby('Date').agg(aggregation_methods).reset_index().sort_values(by='Date')
        cluid_group = normalize_dates(cluid_group)

        # Determine the full date range for reindexing
        full_date_range = pd.date_range(start=cluid_group['Normalized Date'].min(), end=cluid_group['Normalized Date'].max())

        # Group by year and store each year's smoothed data
        time_series_data = {}
        original_time_series_data = {}
        for year, group in cluid_group.groupby(cluid_group['Date'].dt.year):
            original_group = group
            smoothed_group = smooth_data(group.copy())

            time_series_data[year] = smoothed_group
            original_time_series_data[year] = original_group

        years = list(time_series_data.keys())
        distances = []

        interpolated_smoothed_data = {}
        interpolated_original_data = {}

        for i in range(len(years)):
            ts1 = time_series_data[years[i]]['Smoothed'].to_numpy()
            date_index1 = time_series_data[years[i]]['Normalized Date']
            ts11 = interpolate_series(ts1, date_index1, full_date_range)
            interpolated_smoothed_data[years[i]] = ts11

            original_ts1 = original_time_series_data[years[i]]['mean'].to_numpy()
            interpolated_original_ts1 = interpolate_series(original_ts1, date_index1, full_date_range)
            interpolated_original_data[years[i]] = interpolated_original_ts1

            for j in range(i + 1, len(years)):
                ts2 = time_series_data[years[j]]['Smoothed'].to_numpy()
                date_index2 = time_series_data[years[j]]['Normalized Date']
                ts22 = interpolate_series(ts2, date_index2, full_date_range)
                interpolated_smoothed_data[years[j]] = ts22

                original_ts2 = original_time_series_data[years[j]]['mean'].to_numpy()
                interpolated_original_ts2 = interpolate_series(original_ts2, date_index2, full_date_range)
                interpolated_original_data[years[j]] = interpolated_original_ts2

                eu_dist = euclidean_distance(ts11, ts22)
                dtw_dist = dtw_distance(ts1, ts2)
                sam_angle = spectral_angle_mapper(ts11, ts22)

                distances.append((cluid, years[i], years[j], eu_dist, dtw_dist, np.degrees(sam_angle)))
                print('years[i]', years[i])
                print('years[j]', years[j])
                print('ts11', ts11)
                print('ts22', ts22)
#         # Create a combined plot for interpolated smoothed and original data
#         fig, (ax_smoothed, ax_original) = plt.subplots(2, 1, figsize=(14, 16))

#         for year in sorted(interpolated_smoothed_data.keys()):
#             ax_smoothed.plot(full_date_range, interpolated_smoothed_data[year], label=f'Smoothed {year}', linestyle='--', marker='o')

#         for year in sorted(interpolated_original_data.keys()):
#             ax_original.plot(full_date_range, interpolated_original_data[year], label=f'Original {year}', linestyle='-', marker='o')

#         ax_original.plot(full_date_range, avg_original_curve, label='Average Original', linestyle='-', linewidth=2, color='black')

#         ax_smoothed.set_title(f'Smoothed Interpolated Time Series Data for CLUID {cluid}')
#         ax_smoothed.set_xlabel('Date (Normalized)')
#         ax_smoothed.set_ylabel('Value')
#         ax_smoothed.legend()
#         ax_smoothed.grid(True)

#         ax_original.set_title(f'Original Interpolated Time Series Data for CLUID {cluid}')
#         ax_original.set_xlabel('Date (Normalized)')
#         ax_original.set_ylabel('Value')
#         ax_original.legend()
#         ax_original.grid(True)

#         plt.show()

        # Store the results
        cluid_results_df = pd.DataFrame(distances, columns=['CLUID', 'Year1', 'Year2', 'Euclidean Distance', 'DTW Distance', 'Spectral Angle (Degrees)'])
        all_results.append(cluid_results_df)

    # Combine all results into a single DataFrame
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Extract yield columns for each CLUID
    yield_columns = ['CLUID', YearOne, YearTwo, YearThree]
    yield_data = df[yield_columns].drop_duplicates()

    # Merge yield data into final_results_df
    final_results_with_yield_df = final_results_df.merge(yield_data, on='CLUID', how='left')
    
    print('interpolated_smoothed_data', interpolated_smoothed_data)
    for year in sorted(interpolated_smoothed_data.keys()):
        print(year)
        print(interpolated_smoothed_data.keys())
        print(interpolated_smoothed_data[year])
     # Calculate mean values for each yield group by year
    for year in years:
        print(years)
        print(year)
        
        yearly_data = df[df['Date'].dt.year == year]

        # Group by yield group and Date, then calculate mean values to handle duplicates
        yearly_data = yearly_data.groupby([f'{year}_yield_group', 'Date']).agg({'mean': 'mean'}).reset_index()
        print('yearly_data', yearly_data)
        # Plot the mean values for each yield group
        plt.figure(figsize=(14, 8))
        for group in yearly_data[f'{year}_yield_group'].unique():
            print(group)
            group_data = yearly_data[yearly_data[f'{year}_yield_group'] == group]
            plt.plot(group_data['Date'], group_data['mean'], label=f'{group} Yield Group')

        plt.title(f'Mean NDVI Values for {year} Yield Groups')
        plt.xlabel('Date')
        plt.ylabel('Mean NDVI')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return final_results_with_yield_df





# In[95]:


df[df['2019_yield_group'] == "Low"]


# In[112]:


df1 = df[:233]
comparison03_inter2 = years_NDVI_TS_comparison(df1, '2019_yield', '2021_yield', '2023_yield')


# In[190]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.signal import savgol_filter
from IPython.display import display


def smooth_data(group, iterations=3):
    group = group.sort_values(by='Date')
    smoothed_values = group['mean'].copy()
    
    for _ in range(iterations):
        for i in range(1, len(group) - 2):
            if group.iloc[i]['mean'] < group.iloc[i - 1]['mean'] and group.iloc[i]['mean'] < group.iloc[i + 1]['mean']:
                smoothed_values.iloc[i] = (group.iloc[i - 1]['mean'] + group.iloc[i + 1]['mean']) / 2
            
            # Additional check and adjustment
            if group.iloc[i]['mean'] < group.iloc[i - 1]['mean'] and group.iloc[i + 1]['mean'] <= 1 and group.iloc[i + 2]['mean'] > group.iloc[i]['mean']:
                smoothed_values.iloc[i + 1] = (group.iloc[i + 2]['mean'] + group.iloc[i - 1]['mean']) / 2
        group['mean'] = smoothed_values
    group['Smoothed'] = smoothed_values
    return group


def smooth_data_golay(group, window_length=5, polyorder=2):
    group = group.sort_values(by='Date')
    group['Smoothed'] = savgol_filter(group['mean'], window_length=window_length, polyorder=polyorder)
    return group

# Function to normalize dates
def normalize_dates(group):
    group['Normalized Date'] = group['Date'].apply(lambda x: x.replace(year=2000))
    group['Normalized Date'] = pd.to_datetime(group['Normalized Date'])
    return group

# Define distance functions
def euclidean_distance(ts1, ts2):
    return np.sqrt(np.sum((ts1 - ts2)**2))

def dtw_distance(ts1, ts2):
    distance, paths = dtw.warping_paths(np.array(ts1, dtype=np.double), np.array(ts2, dtype=np.double))
    return distance

def spectral_angle_mapper(ts1, ts2):
    dot_product = np.dot(ts1, ts2)
    norm_ts1 = np.linalg.norm(ts1)
    norm_ts2 = np.linalg.norm(ts2)
    return np.arccos(dot_product / (norm_ts1 * norm_ts2))

# Define aggregation methods dynamically
def get_aggregation_methods(df):
    aggregation_methods = {}
    for col in df.columns:
        if col == 'mean':
            aggregation_methods[col] = 'mean'  # Average for 'mean' column
        elif col != 'Date':
            aggregation_methods[col] = 'first'  # 'First' for all other columns
    return aggregation_methods

def interpolate_series(ts, date_index, full_date_range):
    series = pd.Series(ts, index=date_index)
    series = series.reindex(full_date_range).interpolate(method='linear').ffill().bfill()
    return series.to_numpy()

def years_NDVI_TS_comparison(df, YearOne, YearTwo, YearThree):
    # Ensure the date column is in datetime format
    df['Date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.drop('date', axis=1)
    df = df[df['mean'] >= 0.3]
    
    # Initialize a list to store all results
    all_results = []
    interpolated_smoothed_data_all = {}

    # Loop through each CLUID
    for cluid, cluid_group in df.groupby('CLUID'):
#         print('cluid', cluid)
#         print('cluid_group', cluid_group)
        interpolated_smoothed_data_all[cluid] = {}
        # Generate the aggregation dictionary based on the current group columns
        aggregation_methods = get_aggregation_methods(cluid_group)

        # Group by 'date', applying defined aggregation methods
        cluid_group = cluid_group.groupby('Date').agg(aggregation_methods).reset_index().sort_values(by='Date')
        cluid_group = normalize_dates(cluid_group)

        # Determine the full date range for reindexing
#         print('cluid_group', cluid_group)
        full_date_range = pd.date_range(start=cluid_group['Normalized Date'].min(), end=cluid_group['Normalized Date'].max())
#         full_date_range_all = pd.date_range(start=cluid_group['Date'].min(), end=cluid_group['Date'].max())
#         print('full_date_rangestart', len(full_date_range))
#         print('full_date_range_allstart', len(full_date_range_all))
        
        # Group by year and store each year's smoothed data
        time_series_data = {}
        original_time_series_data = {}
        yield_group_data = {}
        full_date_range_data = {}
        
        for year, group in cluid_group.groupby(cluid_group['Date'].dt.year):
            original_group = group
            smoothed_group = smooth_data(group.copy())

            time_series_data[year] = smoothed_group
            original_time_series_data[year] = original_group
            
            # Extract yield group values for the current year
            yield_group_data[year] = group[f'{year}_yield_group'].iloc[0]
            full_date_range_data[year] = pd.date_range(start=group['Date'].min(), end=group['Date'].max())
            print(full_date_range_data)
        years = list(time_series_data.keys())
        distances = []

        interpolated_smoothed_data = {}
        interpolated_original_data = {}

        for i in range(len(years)):
            ts1 = time_series_data[years[i]]['Smoothed'].to_numpy()
            date_index1 = time_series_data[years[i]]['Normalized Date']
            ts11 = interpolate_series(ts1, date_index1, full_date_range)
            interpolated_smoothed_data[years[i]] = ts11
            
            interpolated_smoothed_data_all[cluid][years[i]] = {
                'interpolated_smoothed': ts11,
                'yield_group': yield_group_data[years[i]],
                'full_date_range': full_date_range_data[years[i]]
            }


            original_ts1 = original_time_series_data[years[i]]['mean'].to_numpy()
            interpolated_original_ts1 = interpolate_series(original_ts1, date_index1, full_date_range)
            interpolated_original_data[years[i]] = interpolated_original_ts1

            for j in range(i + 1, len(years)):
                ts2 = time_series_data[years[j]]['Smoothed'].to_numpy()
                date_index2 = time_series_data[years[j]]['Normalized Date']
                ts22 = interpolate_series(ts2, date_index2, full_date_range)
                interpolated_smoothed_data[years[j]] = ts22
                
                interpolated_smoothed_data_all[cluid][years[j]] = {
                    'interpolated_smoothed': ts22,
                    'yield_group': yield_group_data[years[j]],
                    'full_date_range': full_date_range_data[years[j]]
                }

                original_ts2 = original_time_series_data[years[j]]['mean'].to_numpy()
                interpolated_original_ts2 = interpolate_series(original_ts2, date_index2, full_date_range)
                interpolated_original_data[years[j]] = interpolated_original_ts2

                eu_dist = euclidean_distance(ts11, ts22)
                dtw_dist = dtw_distance(ts1, ts2)
                sam_angle = spectral_angle_mapper(ts11, ts22)

                distances.append((cluid, years[i], years[j], eu_dist, dtw_dist, np.degrees(sam_angle)))

#         # Create a combined plot for interpolated smoothed and original data
#         fig, (ax_smoothed, ax_original) = plt.subplots(2, 1, figsize=(14, 16))

#         for year in sorted(interpolated_smoothed_data.keys()):
#             ax_smoothed.plot(full_date_range, interpolated_smoothed_data[year], label=f'Smoothed {year}', linestyle='--', marker='o')

#         for year in sorted(interpolated_original_data.keys()):
#             ax_original.plot(full_date_range, interpolated_original_data[year], label=f'Original {year}', linestyle='-', marker='o')

#         ax_original.plot(full_date_range, avg_original_curve, label='Average Original', linestyle='-', linewidth=2, color='black')

#         ax_smoothed.set_title(f'Smoothed Interpolated Time Series Data for CLUID {cluid}')
#         ax_smoothed.set_xlabel('Date (Normalized)')
#         ax_smoothed.set_ylabel('Value')
#         ax_smoothed.legend()
#         ax_smoothed.grid(True)

#         ax_original.set_title(f'Original Interpolated Time Series Data for CLUID {cluid}')
#         ax_original.set_xlabel('Date (Normalized)')
#         ax_original.set_ylabel('Value')
#         ax_original.legend()
#         ax_original.grid(True)

#         plt.show()

        # Store the results
        cluid_results_df = pd.DataFrame(distances, columns=['CLUID', 'Year1', 'Year2', 'Euclidean Distance', 'DTW Distance', 'Spectral Angle (Degrees)'])
        all_results.append(cluid_results_df)

        
    print('interpolated_smoothed_data_all', interpolated_smoothed_data_all)    
    # Plotting each year with curves for each yield group
    for year in years:
        plt.figure(figsize=(10, 6))
        for cluid in interpolated_smoothed_data_all:
            data = interpolated_smoothed_data_all[cluid].get(year)
            if data:
                full_date_range = data['full_date_range']
                if data['yield_group'] == 'High':
                    plt.plot(full_date_range, data['interpolated_smoothed'], label=f'{cluid} - High', color='green')
                elif data['yield_group'] == 'Medium':
                    plt.plot(full_date_range, data['interpolated_smoothed'], label=f'{cluid} - Medium', color='orange')
                elif data['yield_group'] == 'Low':
                    plt.plot(full_date_range, data['interpolated_smoothed'], label=f'{cluid} - Low', color='red')
        
        plt.title(f'Interpolated Smoothed Data for {year}')
        plt.xlabel('Date')
        plt.ylabel('Smoothed Value')
        plt.legend()
        plt.show()
        
    # Combine all results into a single DataFrame
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Extract yield columns for each CLUID
    yield_columns = ['CLUID', YearOne, YearTwo, YearThree]
    yield_data = df[yield_columns].drop_duplicates()

    # Merge yield data into final_results_df
    final_results_with_yield_df = final_results_df.merge(yield_data, on='CLUID', how='left')
    
    print('interpolated_smoothed_data_all', interpolated_smoothed_data_all)
#     for year in sorted(interpolated_smoothed_data.keys()):
#         print(year)
#         print(interpolated_smoothed_data.keys())
#         print(interpolated_smoothed_data[year])
     # Calculate mean values for each yield group by year
#     for year in years:
#         print(years)
#         print(year)
        
#         yearly_data = df[df['Date'].dt.year == year]

#         # Group by yield group and Date, then calculate mean values to handle duplicates
#         yearly_data = yearly_data.groupby([f'{year}_yield_group', 'Date']).agg({'mean': 'mean'}).reset_index()
#         print('yearly_data', yearly_data)
#         # Plot the mean values for each yield group
#         plt.figure(figsize=(14, 8))
#         for group in yearly_data[f'{year}_yield_group'].unique():
#             print(group)
#             group_data = yearly_data[yearly_data[f'{year}_yield_group'] == group]
#             plt.plot(group_data['Date'], group_data['mean'], label=f'{group} Yield Group')

#         plt.title(f'Mean NDVI Values for {year} Yield Groups')
#         plt.xlabel('Date')
#         plt.ylabel('Mean NDVI')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
    return final_results_with_yield_df





# In[189]:


df1 = df[:233]
comparison03_inter2 = years_NDVI_TS_comparison(df1, '2019_yield', '2021_yield', '2023_yield')


# # new way

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Sample custom smoothing function (replace with your own)
def custom_smooth_function(values):
    # Example smoothing: simple moving average with window size 3
    return values.rolling(window=3, min_periods=1).mean()

def smooth_data(values, iterations=3):
    smoothed_values = values.copy()
    
    for _ in range(iterations):
        new_smoothed_values = smoothed_values.copy()
        for i in range(1, len(smoothed_values) - 2):
            if smoothed_values.iloc[i] < smoothed_values.iloc[i - 1] and smoothed_values.iloc[i] < smoothed_values.iloc[i + 1]:
                new_smoothed_values.iloc[i] = (smoothed_values.iloc[i - 1] + smoothed_values.iloc[i + 1]) / 2
            
            # Additional check and adjustment
            if smoothed_values.iloc[i] < smoothed_values.iloc[i - 1] and smoothed_values.iloc[i + 1] <= 1 and smoothed_values.iloc[i + 2] > smoothed_values.iloc[i]:
                new_smoothed_values.iloc[i + 1] = (smoothed_values.iloc[i + 2] + smoothed_values.iloc[i - 1]) / 2
        smoothed_values = new_smoothed_values
    return smoothed_values

# Load your DataFrame
path = r"C:\Users\User\Documents\Yields_project\Proag_Yields\proAg_yield_pw_units_07012024\years_comparison_PA_NDVI_TS_23_21_19_from_all_history.csv"

df = pd.read_csv(path)
df = df.loc[(df['2019_yield'] != 0) & (df['2021_yield'] != 0) & (df['2023_yield'] != 0)]
df = df.rename(columns={'PWId': 'CLUID'})

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['date'], dayfirst=True)

# Function to categorize yields
def categorize_yield(yield_value):
    if yield_value < 180:
        return 'Low'
    elif 150 <= yield_value <= 220:
        return 'Medium'
    else:
        return 'High'

# Apply the function to each yield column and create a new group column
for column in df.columns:
    if 'yield' in column:
        df[column + '_group'] = df[column].apply(categorize_yield)

        # Function to apply the categorization based on the Date column
def get_yield_group(row):
    year = row['Date'].year
    yield_value = row[f'{year}_yield']
    return categorize_yield(yield_value)

# Apply the function to each row and create a new 'yield_group' column
df['Date_year_yield_group'] = df.apply(get_yield_group, axis=1)
print(df)
# Step 1: Group by `CLUID` and `Date` to calculate the mean of the `mean` column for duplicates
# Include all necessary columns in the aggregation
agg_dict = {col: 'first' for col in df.columns if col not in ['CLUID', 'Date', 'mean']}
agg_dict['mean'] = 'mean'

df_grouped = df.groupby(['CLUID', 'Date']).agg(agg_dict).reset_index()

# Step 2: Smooth the mean values by your custom function for each `CLUID` and year
df_grouped['Year'] = df_grouped['Date'].dt.year
smoothed_means = []

for cluid, group in df_grouped.groupby('CLUID'):
    for year, year_group in group.groupby('Year'):
        year_group = year_group.sort_values('Date')
        year_group['smoothed_mean'] = smooth_data(year_group['mean'])
        smoothed_means.append(year_group)

df_smoothed = pd.concat(smoothed_means)

# Step 4: Interpolate the mean values over the date range for each year
def interpolate_series(ts, date_index, full_date_range):
    series = pd.Series(ts.values, index=date_index)
    series = series.reindex(full_date_range)
    series = series.interpolate(method='linear')
    series = series.ffill().bfill()
    return series.to_numpy()

def interpolate_values(group, year):
    date_range = pd.date_range(start=f'{year}-05-01', end=f'{year}-11-01')
    interpolated_means = interpolate_series(group['smoothed_mean'], group['Date'], date_range)
    interpolated_df = pd.DataFrame({'Date': date_range, 'interpolated_mean': interpolated_means})
    for col in group.columns:
        if col not in ['Date', 'mean', 'smoothed_mean', 'Year']:
            interpolated_df[col] = group[col].iloc[0]
    interpolated_df['Year'] = year
    return interpolated_df

interpolated_means = []

for cluid, group in df_smoothed.groupby('CLUID'):
    for year, year_group in group.groupby('Year'):
        interpolated_means.append(interpolate_values(year_group, year))

df_interpolated = pd.concat(interpolated_means)
# df_interpolated.to_csv(r"C:\Users\User\Downloads\ckeck.csv")

# Step 5: Plot each year's mean values of high, medium, and low yields
years = df_interpolated['Year'].unique()
yield_groups = ['Low', 'Medium', 'High']

fig, axes = plt.subplots(len(years), 1, figsize=(12, len(years) * 5), sharex=False)

for i, year in enumerate(years):
    ax = axes[i]
    year_data = df_interpolated[df_interpolated['Year'] == year]
    for yield_group in yield_groups:
        group_data = year_data[year_data[f'{year}_yield_group'] == yield_group]
        if not group_data.empty:
            group_mean = group_data.groupby('Date')['interpolated_mean'].mean()
            ax.plot(group_mean.index, group_mean.values, label=yield_group)
    
    ax.set_title(f'Mean Values of Yield Groups in {year}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Value')
    ax.legend()
    
    # Set x-axis to only show dates from the relevant year
    date_range = pd.date_range(start=f'{year}-05-01', end=f'{year}-11-01', freq='MS')
    ax.set_xlim([pd.Timestamp(f'{year}-05-01'), pd.Timestamp(f'{year}-11-01')])
    ax.set_xticks(date_range)
    ax.set_xticklabels(date_range.strftime('%Y-%m-%d'), rotation=45)

plt.tight_layout()
plt.show()


# ## by month

# In[86]:


df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
# print(df_interpolated.info())
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])]
print('df_2019_2021', df_2019_2021.head())

# def get_yield_group(row):
#     if row['Year'] == 2019:
#         return row['2019_yield_group']
#     elif row['Year'] == 2021:
#         return row['2021_yield_group']
#     else:
#         return None
mean_2019_2021 = df_2019_2021.groupby(['Date', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()
print('mean_2019_2021', mean_2019_2021)
mean_2019_2021['normalized_dates'] = mean_2019_2021['Date'].apply(lambda x: x.replace(year=2023))
mean_2019_2021['Month'] = mean_2019_2021['Date'].dt.month
mean_2019_2021_combined = mean_2019_2021.groupby(['Month', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()
print('mean_2019_2021_combined', mean_2019_2021_combined)
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021_combined[mean_2019_2021_combined['Date_year_yield_group'] == yield_group]
        ax.plot(group_data['Month'], group_data['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Plot 2023 CLUID data
    ax.plot(cluid_data['Date'].dt.month, cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[114]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Assuming df_interpolated is already defined and loaded

# Convert 'Date' to datetime
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])

# Filter data for years 2019 and 2021
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])]
print('df_2019_2021', df_2019_2021.head())

# Group by 'Date' and 'Date_year_yield_group' and calculate the mean
mean_2019_2021 = df_2019_2021.groupby(['Date', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()
print('mean_2019_2021', mean_2019_2021)

# Normalize dates to the same year for comparison
mean_2019_2021['normalized_dates'] = mean_2019_2021['Date'].apply(lambda x: x.replace(year=2023))
mean_2019_2021['Month'] = mean_2019_2021['Date'].dt.month

# Group by 'Month' and 'Date_year_yield_group' and calculate the mean
mean_2019_2021_combined = mean_2019_2021.groupby(['Month', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()
print('mean_2019_2021_combined', mean_2019_2021_combined)

# Get unique CLUIDs for the year 2023
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

# Calculate Euclidean distance and plot data
for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    distances = {}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021_combined[mean_2019_2021_combined['Date_year_yield_group'] == yield_group]
        ax.plot(group_data['Month'], group_data['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

        # Calculate Euclidean distance
        cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
        cluid_monthly_data = cluid_data.groupby(cluid_data['Date'].dt.month)['interpolated_mean'].mean()
        group_monthly_data = group_data.set_index('Month')['interpolated_mean']
        
        # Align the data for distance calculation
        aligned_cluid_data = cluid_monthly_data.reindex(group_monthly_data.index, fill_value=0)
        distances[yield_group] = euclidean(aligned_cluid_data, group_monthly_data)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Plot 2023 CLUID data
    ax.plot(cluid_data['Date'].dt.month, cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    # Print the Euclidean distances
    for yield_group, distance in distances.items():
        print(f"Euclidean distance between 2023 CLUID {cluid} and 2019-2021 {yield_group} group: {distance:.2f}")

    # Find the most similar yield group
    most_similar_group = min(distances, key=distances.get)
    print(f"The 2023 CLUID {cluid} is most similar to the 2019-2021 {most_similar_group} group.")

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Month')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[63]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])]
print('df_2019_2021', df_2019_2021)
# Use 'yield_group' to unify the yield group columns for 2019 and 2021
df_2019_2021['yield_group'] = df_2019_2021.apply(lambda row: row['2019_yield_group'] if '2019_yield_group' in row else row['2021_yield_group'], axis=1)

# Calculate the mean interpolated_mean for each yield group by date
mean_2019_2021 = df_2019_2021.groupby(['Date', 'yield_group'])['interpolated_mean'].mean().reset_index()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['yield_group'] == yield_group]
        group_data['normalized_dates'] = group_data['Date'].apply(lambda x: x.replace(year=2023))
        group_data_combined = group_data.groupby(['normalized_dates'])['interpolated_mean'].mean().reset_index()
        ax.plot(group_data_combined['normalized_dates'], group_data_combined['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Plot 2023 CLUID data
    ax.plot(cluid_data['Date'], cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[51]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())
# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])]

# Use 'yield_group' to unify the yield group columns for 2019 and 2021
df_2019_2021['yield_group'] = df_2019_2021.apply(lambda row: row['2019_yield_group'] if '2019_yield_group' in row else row['2021_yield_group'], axis=1)

# Calculate the mean interpolated_mean for each yield group by date
mean_2019_2021 = df_2019_2021.groupby(['Date', 'yield_group'])['interpolated_mean'].mean().reset_index()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['yield_group'] == yield_group]
        group_data['normalized_dates'] = group_data['Date'].apply(lambda x: x.replace(year=2023))
        group_data_combined = group_data.groupby(['normalized_dates'])['interpolated_mean'].mean().reset_index()
        ax.plot(group_data_combined['normalized_dates'].dt.month, group_data_combined['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Plot 2023 CLUID data
    ax.plot(cluid_data['Date'].dt.month, cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ## by Date

# In[87]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())
# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])]

# Calculate the mean interpolated_mean for each yield group by date
mean_2019_2021 = df_2019_2021.groupby(['Date', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['Date_year_yield_group'] == yield_group]
        group_data['normalized_dates'] = group_data['Date'].apply(lambda x: x.replace(year=2023))
        group_data_combined = group_data.groupby(['normalized_dates'])['interpolated_mean'].mean().reset_index()
        ax.plot(group_data_combined['normalized_dates'], group_data_combined['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Plot 2023 CLUID data
    ax.plot(cluid_data['Date'], cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[120]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())

# Filter data for years 2019 and 2021
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])]

# Calculate the mean interpolated_mean for each yield group by date
mean_2019_2021 = df_2019_2021.groupby(['Date', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()

# Get unique CLUIDs for the year 2023
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

# Calculate Euclidean distance and plot data
for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    distances = {}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['Date_year_yield_group'] == yield_group]
        group_data['normalized_dates'] = group_data['Date'].apply(lambda x: x.replace(year=2023))
        group_data_combined = group_data.groupby(['normalized_dates'])['interpolated_mean'].mean().reset_index()
        ax.plot(group_data_combined['normalized_dates'], group_data_combined['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

        # Calculate Euclidean distance
        cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)]
        cluid_data['normalized_dates'] = cluid_data['Date'].apply(lambda x: x.replace(year=2023))
        cluid_combined = cluid_data.groupby(['normalized_dates'])['interpolated_mean'].mean().reset_index()

        # Align the data for distance calculation
        merged_data = pd.merge(cluid_combined, group_data_combined, on='normalized_dates', suffixes=('_2023', '_2019_2021'))
        distances[yield_group] = euclidean(merged_data['interpolated_mean_2023'], merged_data['interpolated_mean_2019_2021'])

    # Determine the yield group of the CLUID in 2023
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Plot 2023 CLUID data
    ax.plot(cluid_combined['normalized_dates'], cluid_combined['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    # Print the Euclidean distances
    for yield_group, distance in distances.items():
        print(f"Euclidean distance between 2023 CLUID {cluid} and 2019-2021 {yield_group} group: {distance:.2f}")

    # Find the most similar yield group
    most_similar_group = min(distances, key=distances.get)
    print(f"The 2023 CLUID {cluid} is most similar to the 2019-2021 {most_similar_group} group.")

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ## by 0.3 days

# In[88]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])].copy()

# Function to normalize dates to 2023
def normalize_dates(df, year=2023):
    df.loc[:, 'normalized_dates'] = df['Date'].apply(lambda x: x.replace(year=year))
    return df

# Normalize dates for 2019-2021 data
df_2019_2021 = normalize_dates(df_2019_2021)

# Calculate the mean interpolated_mean for each yield group by normalized date
mean_2019_2021 = df_2019_2021.groupby(['normalized_dates', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()

# Function to find the reference date when interpolated_mean crosses 0.3
def find_reference_date(data):
    return data[data['interpolated_mean'] >= 0.3].iloc[0]['normalized_dates'] if not data[data['interpolated_mean'] >= 0.3].empty else None

# Find reference dates for each yield group in mean data
reference_dates = mean_2019_2021.groupby('Date_year_yield_group').apply(lambda x: find_reference_date(x)).to_dict()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['Date_year_yield_group'] == yield_group].copy()
        reference_date = reference_dates.get(yield_group)
        if reference_date:
            group_data.loc[:, 'days_from_ref'] = (group_data['normalized_dates'] - reference_date).dt.days
            group_data_combined = group_data.groupby('days_from_ref')['interpolated_mean'].mean().reset_index()
            ax.plot(group_data_combined['days_from_ref'], group_data_combined['interpolated_mean'], 
                    label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)].copy()
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Normalize dates for the CLUID data in 2023
    cluid_data = normalize_dates(cluid_data)

    # Find reference date for the CLUID in 2023
    reference_date_2023 = find_reference_date(cluid_data)
    if reference_date_2023:
        cluid_data.loc[:, 'days_from_ref'] = (cluid_data['normalized_dates'] - reference_date_2023).dt.days
        # Plot 2023 CLUID data
        ax.plot(cluid_data['days_from_ref'], cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Days from 0.3 Value')
    ax.set_ylabel('Interpolated Mean')
    ax.set_xlim(left=0)  # Start x-axis from 0
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[121]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])].copy()

# Function to normalize dates to 2023
def normalize_dates(df, year=2023):
    df.loc[:, 'normalized_dates'] = df['Date'].apply(lambda x: x.replace(year=year))
    return df

# Normalize dates for 2019-2021 data
df_2019_2021 = normalize_dates(df_2019_2021)

# Calculate the mean interpolated_mean for each yield group by normalized date
mean_2019_2021 = df_2019_2021.groupby(['normalized_dates', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()

# Function to find the reference date when interpolated_mean crosses 0.3
def find_reference_date(data):
    return data[data['interpolated_mean'] >= 0.3].iloc[0]['normalized_dates'] if not data[data['interpolated_mean'] >= 0.3].empty else None

# Find reference dates for each yield group in mean data
reference_dates = mean_2019_2021.groupby('Date_year_yield_group').apply(lambda x: find_reference_date(x)).to_dict()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

# Calculate Euclidean distance and plot data
for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    distances = {}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['Date_year_yield_group'] == yield_group].copy()
        reference_date = reference_dates.get(yield_group)
        if reference_date:
            group_data.loc[:, 'days_from_ref'] = (group_data['normalized_dates'] - reference_date).dt.days
            group_data_combined = group_data.groupby('days_from_ref')['interpolated_mean'].mean().reset_index()
            ax.plot(group_data_combined['days_from_ref'], group_data_combined['interpolated_mean'], 
                    label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)
            
            # Calculate Euclidean distance
            cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)].copy()
            cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

            # Normalize dates for the CLUID data in 2023
            cluid_data = normalize_dates(cluid_data)

            # Find reference date for the CLUID in 2023
            reference_date_2023 = find_reference_date(cluid_data)
            if reference_date_2023:
                cluid_data.loc[:, 'days_from_ref'] = (cluid_data['normalized_dates'] - reference_date_2023).dt.days
                cluid_combined = cluid_data.groupby('days_from_ref')['interpolated_mean'].mean().reset_index()

                # Align the data for Euclidean distance calculation
                merged_data = pd.merge(cluid_combined, group_data_combined, on='days_from_ref', suffixes=('_2023', '_2019_2021'))
                distances[yield_group] = euclidean(merged_data['interpolated_mean_2023'], merged_data['interpolated_mean_2019_2021'])

    # Plot 2023 CLUID data
    if reference_date_2023:
        ax.plot(cluid_combined['days_from_ref'], cluid_combined['interpolated_mean'], 
                label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    # Print the Euclidean distances
    for yield_group, distance in distances.items():
        print(f"Euclidean distance between 2023 CLUID {cluid} and 2019-2021 {yield_group} group: {distance:.2f}")

    # Find the most similar yield group
    if distances:
        most_similar_group = min(distances, key=distances.get)
        print(f"The 2023 CLUID {cluid} is most similar to the 2019-2021 {most_similar_group} group.")

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Days from 0.3 Value')
    ax.set_ylabel('Interpolated Mean')
    ax.set_xlim(left=0)  # Start x-axis from 0
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ## August values

# In[89]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())

# Filter for August dates
df_interpolated = df_interpolated[df_interpolated['Date'].dt.month == 8]

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated[df_interpolated['Year'].isin([2019, 2021])].copy()

# Function to normalize dates to 2023
def normalize_dates(df, year=2023):
    df['normalized_dates'] = df['Date'].apply(lambda x: x.replace(year=year))
    return df

# Normalize dates for 2019-2021 data
df_2019_2021 = normalize_dates(df_2019_2021)

# Calculate the mean interpolated_mean for each yield group by normalized date
mean_2019_2021 = df_2019_2021.groupby(['normalized_dates', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data
cluid_2023 = df_interpolated[df_interpolated['Year'] == 2023]['CLUID'].unique()

for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data = mean_2019_2021[mean_2019_2021['Date_year_yield_group'] == yield_group].copy()
        group_data_combined = group_data.groupby(['normalized_dates'])['interpolated_mean'].mean().reset_index()
        ax.plot(group_data_combined['normalized_dates'], group_data_combined['interpolated_mean'], 
                label=f'2019-2021 {yield_group} Mean', color=colors[yield_group], linewidth=2)

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated[(df_interpolated['Year'] == 2023) & (df_interpolated['CLUID'] == cluid)].copy()
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Normalize dates for the CLUID data in 2023
    cluid_data = normalize_dates(cluid_data)

    # Plot 2023 CLUID data
    ax.plot(cluid_data['Date'], cluid_data['interpolated_mean'], label=f'2023 CLUID {cluid} ({cluid_yield_group})', linestyle='--', color='black', linewidth=2)

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group')
    ax.set_xlabel('Date')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ### August mean value

# In[90]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())

# Filter for August 1st to August 15th dates
df_interpolated_august = df_interpolated[(df_interpolated['Date'].dt.month == 8)]

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated_august[df_interpolated_august['Year'].isin([2019, 2021])].copy()


# Calculate the mean interpolated_mean for each yield group for the first 15 days of August
mean_2019_2021_august = df_2019_2021.groupby('Date_year_yield_group')['interpolated_mean'].mean().reset_index()
print('mean_2019_2021_august', mean_2019_2021_august)
# Step 6: Plot each 2023 CLUID with 2019-2021 mean data for August 1st to 15th
cluid_2023 = df_interpolated_august[df_interpolated_august['Year'] == 2023]['CLUID'].unique()


for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data_mean = mean_2019_2021_august[mean_2019_2021_august['Date_year_yield_group'] == yield_group]
        ax.plot('August 1st to 15th', group_data_mean['interpolated_mean'], 'o', color=colors[yield_group], label=f'2019-2021 {yield_group} Mean')

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated_august[(df_interpolated_august['Year'] == 2023) & (df_interpolated_august['CLUID'] == cluid)].copy()
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Calculate mean for the 2023 CLUID data in August 1st to 15th
    cluid_mean_august = cluid_data['interpolated_mean'].mean()

    # Plot 2023 CLUID data mean for August 1st to 15th
    ax.plot('August 1st to 15th', cluid_mean_august, 'o', color='black', label=f'2023 CLUID {cluid} ({cluid_yield_group})')

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group (August 1st to 15th)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ### August 1st until 15th mean value

# In[20]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())

# Filter for August 1st to August 15th dates
df_interpolated_august = df_interpolated[(df_interpolated['Date'].dt.month == 8) & (df_interpolated['Date'].dt.day <= 15)]

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated_august[df_interpolated_august['Year'].isin([2019, 2021])].copy()


# Calculate the mean interpolated_mean for each yield group for the first 15 days of August
mean_2019_2021_august = df_2019_2021.groupby('Date_year_yield_group')['interpolated_mean'].mean().reset_index()

# Step 6: Plot each 2023 CLUID with 2019-2021 mean data for August 1st to 15th
cluid_2023 = df_interpolated_august[df_interpolated_august['Year'] == 2023]['CLUID'].unique()


for cluid in cluid_2023:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data_mean = mean_2019_2021_august[mean_2019_2021_august['Date_year_yield_group'] == yield_group]
        ax.plot('August 1st to 15th', group_data_mean['interpolated_mean'], 'o', color=colors[yield_group], label=f'2019-2021 {yield_group} Mean')

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated_august[(df_interpolated_august['Year'] == 2023) & (df_interpolated_august['CLUID'] == cluid)].copy()
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Calculate mean for the 2023 CLUID data in August 1st to 15th
    cluid_mean_august = cluid_data['interpolated_mean'].mean()

    # Plot 2023 CLUID data mean for August 1st to 15th
    ax.plot('August 1st to 15th', cluid_mean_august, 'o', color='black', label=f'2023 CLUID {cluid} ({cluid_yield_group})')

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Mean by Yield Group (August 1st to 15th)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.tight_layout()
    plt.show()


# In[97]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
print(df_interpolated.info())

# Filter for August 1st to August 15th dates
df_interpolated_august = df_interpolated[(df_interpolated['Date'].dt.month == 8) & (df_interpolated['Date'].dt.day <= 15)]

# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated_august[df_interpolated_august['Year'].isin([2019, 2021])].copy()


# Calculate the mean interpolated_mean for each yield group for the first 15 days of August
mean_2019_2021_august = df_2019_2021.groupby('Date_year_yield_group')['interpolated_mean'].mean().reset_index()
print('mean_2019_2021_august', mean_2019_2021_august)
# Step 6: Calculate mean for each CLUID in 2023 for the first 15 days of August
cluid_2023_means = df_interpolated_august[df_interpolated_august['Year'] == 2023].groupby(['CLUID', '2023_yield_group'])['interpolated_mean'].mean().reset_index()
cluid_2023_means.columns = ['CLUID', '2023_yield_group', 'mean_interpolated_mean']

# Find the closest yield group for each CLUID in 2023
def find_closest_yield_group(mean_value, mean_2019_2021):
    differences = mean_2019_2021['interpolated_mean'] - mean_value
    closest_index = differences.abs().idxmin()
    return mean_2019_2021.loc[closest_index, 'Date_year_yield_group']

cluid_2023_means['adjusted_yield_group'] = cluid_2023_means['mean_interpolated_mean'].apply(
    lambda x: find_closest_yield_group(x, mean_2019_2021_august))
print(cluid_2023_means)
cluid_2023_means.to_csv(r"C:\Users\User\Downloads\ckeck_match.csv")

# Plot each 2023 CLUID with 2019-2021 mean data for August 1st to 15th
for cluid in cluid_2023_means['CLUID']:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data_mean = mean_2019_2021_august[mean_2019_2021_august['Date_year_yield_group'] == yield_group]
        ax.plot('August 1st to 15th', group_data_mean['interpolated_mean'], 'o', color=colors[yield_group], label=f'2019-2021 {yield_group} Mean')

    # Get the mean, 2023 yield group, and adjusted yield group for the 2023 CLUID
    cluid_mean_august = cluid_2023_means[cluid_2023_means['CLUID'] == cluid]['mean_interpolated_mean'].values[0]
    original_yield_group = cluid_2023_means[cluid_2023_means['CLUID'] == cluid]['2023_yield_group'].values[0]
    adjusted_yield_group = cluid_2023_means[cluid_2023_means['CLUID'] == cluid]['adjusted_yield_group'].values[0]

    # Plot 2023 CLUID data mean for August 1st to 15th
    ax.plot('August 1st to 15th', cluid_mean_august, 'o', color='black', label=f'2023 CLUID {cluid} ({original_yield_group}) Adjusted to {adjusted_yield_group}')

    ax.set_title(f'2023 CLUID {cluid} ({original_yield_group}) Adjusted Yield Group: {adjusted_yield_group}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.tight_layout()
    plt.show()


# In[123]:


# Calculate the percentage match where 2023_yield_group equals "Low" and adjusted_yield_group equals "Low" or "Medium"
total_low_rows = len(cluid_2023_means[cluid_2023_means['2023_yield_group'] == 'Low'])
matching_low_rows = len(cluid_2023_means[(cluid_2023_means['2023_yield_group'] == 'Low') & (cluid_2023_means['adjusted_yield_group'].isin(['Low', 'Medium']))])
percentage_match_low = (matching_low_rows / total_low_rows) * 100 if total_low_rows > 0 else 0

print(f"Percentage of rows where 2023_yield_group equals 'Low' and adjusted_yield_group equals 'Low' or 'Medium': {percentage_match_low:.2f}%")

# Calculate the percentage match where 2023_yield_group equals "Low" and adjusted_yield_group equals "Low" 
total_low_rows2 = len(cluid_2023_means[cluid_2023_means['2023_yield_group'] == 'Low'])
matching_low_rows2 = len(cluid_2023_means[(cluid_2023_means['adjusted_yield_group'] == 'Low') & (cluid_2023_means['2023_yield_group'] == 'Low')])
percentage_match_low2 = (matching_low_rows2 / total_low_rows2) * 100 if total_low_rows2 > 0 else 0

print(f"Percentage of rows where 2023_yield_group equals 'Low' and adjusted_yield_group equals 'Low': {percentage_match_low2:.2f}%")

# Creating a confusion matrix with totals
conf_matrix = pd.crosstab(cluid_2023_means['2023_yield_group'], cluid_2023_means['adjusted_yield_group'], 
                          rownames=['2023 Yield Group'], colnames=['Adjusted Yield Group'], margins=True)

# Calculating overall accuracy
total_correct = sum(conf_matrix.at[x, x] for x in conf_matrix.columns[:-1])  # Exclude the 'All' column
total_observations = conf_matrix.at['All', 'All']
accuracy = (total_correct / total_observations) * 100 if total_observations > 0 else 0

# Calculating accuracy specifically for the "Low" group
low_group_correct = conf_matrix.at['Low', 'Low']
low_group_total = conf_matrix.at['Low', 'All']
low_group_accuracy = (low_group_correct / low_group_total) * 100 if low_group_total > 0 else 0

print(conf_matrix)
print(f"Overall Accuracy: {accuracy:.2f}%")
print(f"Accuracy for 'Low' group: {low_group_accuracy:.2f}%")


# ### Max value in August

# In[112]:


# Ensure Date is in datetime format
df_interpolated['Date'] = pd.to_datetime(df_interpolated['Date'])
# print(df_interpolated.info())

# Filter for August 1st to August 15th dates
df_interpolated_august = df_interpolated[(df_interpolated['Date'].dt.month == 8) & (df_interpolated['Date'].dt.day <= 15)]
print('df_interpolated_august', df_interpolated_august)


# Function to normalize dates to 2023
def normalize_dates(df, year=2023):
    df['normalized_dates'] = df['Date'].apply(lambda x: x.replace(year=year))
    return df


# Normalize the dates to match the 2023 dates
df_2019_2021 = df_interpolated_august[df_interpolated_august['Year'].isin([2019, 2021])].copy()
df_2019_2021 = normalize_dates(df_2019_2021)

# Calculate the maximum interpolated_mean for each yield group for August
mean_2019_2021 = df_2019_2021.groupby(['normalized_dates', 'Date_year_yield_group'])['interpolated_mean'].mean().reset_index()

max_2019_2021_august = mean_2019_2021.groupby('Date_year_yield_group')['interpolated_mean'].max().reset_index()
print('max_2019_2021_august', max_2019_2021_august)
# Calculate the maximum interpolated_mean for each 2023 CLUID for August
cluid_2023_max = df_interpolated_august[df_interpolated_august['Year'] == 2023].groupby('CLUID')['interpolated_mean'].max().reset_index()
print('cluid_2023_max1', cluid_2023_max)

# Add the 2023_yield_group column
cluid_2023_yield_groups = df_interpolated_august[df_interpolated_august['Year'] == 2023][['CLUID', '2023_yield_group']].drop_duplicates()
cluid_2023_max = cluid_2023_max.merge(cluid_2023_yield_groups, on='CLUID')
cluid_2023_max.rename(columns={'interpolated_mean': 'max_interpolated_mean'}, inplace=True)
print('cluid_2023_max2', cluid_2023_max)

# Define a function to find the closest yield group
def find_closest_yield_group(max_value, max_2019_2021):
    differences = max_2019_2021['interpolated_mean'] - max_value
    closest_index = differences.abs().idxmin()
    return max_2019_2021.loc[closest_index, 'Date_year_yield_group']

# Find the closest yield group for each 2023 CLUID
cluid_2023_max['adjusted_yield_group'] = cluid_2023_max['max_interpolated_mean'].apply(
    lambda x: find_closest_yield_group(x, max_2019_2021_august))
print('cluid_2023_max3', cluid_2023_max)

# Plot each 2023 CLUID with 2019-2021 maximum data for August
for cluid in cluid_2023_max['CLUID']:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot maximum 2019-2021 data for High, Medium, and Low yield groups
    colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for yield_group in ['High', 'Medium', 'Low']:
        group_data_max = max_2019_2021_august[max_2019_2021_august['Date_year_yield_group'] == yield_group]
        ax.plot('August 1st to 15th', group_data_max['interpolated_mean'], 'o', color=colors[yield_group], label=f'2019-2021 {yield_group} Max')

    # Determine the yield group of the CLUID in 2023
    cluid_data = df_interpolated_august[(df_interpolated_august['Year'] == 2023) & (df_interpolated_august['CLUID'] == cluid)].copy()
    cluid_yield_group = cluid_data.iloc[0]['2023_yield_group']

    # Calculate maximum for the 2023 CLUID data in August 1st to 15th
    cluid_max_august = cluid_data['interpolated_mean'].max()

    # Plot 2023 CLUID data maximum for August
    ax.plot('August 1st to 15th', cluid_max_august, 'o', color='black', label=f'2023 CLUID {cluid} ({cluid_yield_group})')

    ax.set_title(f'2023 CLUID {cluid} ({cluid_yield_group}) vs. 2019-2021 Max by Yield Group (August 1st to 15th)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Interpolated Mean')
    ax.legend()
    plt.tight_layout()
    plt.show()


# In[122]:


# Calculate the percentage match where 2023_yield_group equals "Low" and adjusted_yield_group equals "Low" or "Medium"
total_low_rows = len(cluid_2023_max[cluid_2023_max['2023_yield_group'] == 'Low'])
matching_low_rows = len(cluid_2023_max[(cluid_2023_max['2023_yield_group'] == 'Low') & (cluid_2023_max['adjusted_yield_group'].isin(['Low', 'Medium']))])
percentage_match_low = (matching_low_rows / total_low_rows) * 100 if total_low_rows > 0 else 0

print(f"Percentage of rows where 2023_yield_group equals 'Low' and adjusted_yield_group equals 'Low' or 'Medium': {percentage_match_low:.2f}%")

# Creating a confusion matrix with totals
conf_matrix = pd.crosstab(cluid_2023_max['2023_yield_group'], cluid_2023_max['adjusted_yield_group'], 
                          rownames=['2023 Yield Group'], colnames=['Adjusted Yield Group'], margins=True)

# Calculating overall accuracy
total_correct = sum(conf_matrix.at[x, x] for x in conf_matrix.columns[:-1])  # Exclude the 'All' column
total_observations = conf_matrix.at['All', 'All']
accuracy = (total_correct / total_observations) * 100 if total_observations > 0 else 0

# Calculating accuracy specifically for the "Low" group
low_group_correct = conf_matrix.at['Low', 'Low']
low_group_total = conf_matrix.at['Low', 'All']
low_group_accuracy = (low_group_correct / low_group_total) * 100 if low_group_total > 0 else 0

print(conf_matrix)
print(f"Overall Accuracy: {accuracy:.2f}%")
print(f"Accuracy for 'Low' group: {low_group_accuracy:.2f}%")


# In[ ]:




