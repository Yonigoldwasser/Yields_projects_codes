#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import os


# In[2]:


RF_data = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")
RF_data.info()


# In[3]:


RF_data1 = RF_data[RF_data['max_index_red']>0]
RF_data1 = RF_data1.dropna()
RF_data1 = RF_data1.sort_values(by='2023_yield')
RF_data1.iloc[:20,:10]


# In[4]:


# List of names to be removed
names_to_remove = ['DBAEE44D-8DA1-40A9-8B3F-FDD47986CE24 ', '098FF92E-741C-40E5-9F72-9809A20FEDA3', '9F352864-D1B9-43A3-BD98-8242FAEA5B43',
                   '016B6B76-B0D6-4E75-9EE9-66919EE8F449', 'F89ED56B-A560-4F84-BE1D-EEE8C8B790A1', '0D050EBA-6B71-426D-9EE8-82D175D6CC00',
                   '62B17017-BBD4-408C-933B-B80FBF5D9649', '30D5A228-EB1C-4067-A5C4-742D0C13248E', 'CA364A07-D39D-4181-9F8A-E21560389F7C',
                   '3ADC01B2-A1C8-4B8A-ABE6-6CA2F8C74F7D','DBAEE44D-8DA1-40A9-8B3F-FDD47986CE24', '3603790d-8744-48a0-97d1-fd566aa88266',
                   '744ABCEB-4D5D-4BDE-8DBD-F1381E7B6773', '95be306b-dc44-437e-b662-3538966cb7cf', '6147CA90-EFE6-4C01-BEE7-E8F955ABCA29']

# Remove rows where 'name' column values are in the names_to_remove list
RF_data1 = RF_data1[~RF_data1['commonland'].isin(names_to_remove)]

# Verify the rows are removed
RF_data1


# # RF

# In[5]:


# RF_data1 = RF_data1.drop(columns='state')

# Prepare data
X = RF_data1.drop(['state','2023_yield', 'yield_group', 'commonland'], axis=1).copy()
y = RF_data1['2023_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)
print("Number of predictions made:", len(y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs. Predicted Yield')

# Add trendline
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r--")

plt.show()

# Feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting Feature Importances
plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='b')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Relative Importance')
plt.show()


# In[6]:


# Create DataFrame with actual and predicted yields for each 'commonland'
test_data = X_test.copy()
test_data['commonland'] = RF_data1.loc[X_test.index, 'commonland']
test_data['Actual_2023_yield'] = y_test
test_data['Predicted_2023_yield'] = y_pred
test_data = test_data.sort_values(by='Predicted_2023_yield')

# Display the DataFrame
# import ace_tools as tools; tools.display_dataframe_to_user(name="Commonland Yield Predictions", dataframe=test_data)

# Print the DataFrame
test_data.iloc[:20,:]
test_data[test_data['Actual_2023_yield']<50]


# In[7]:


# List to store the results for each state
results = []
RF_data2 = RF_data.dropna()

# Get unique states in the dataset
states = RF_data2['state'].unique()

# List to store the aggregated results
agg_results = []

for state in states:
    # Filter data for the current state
    state_data = RF_data2[RF_data2['state'] == state]
    
    # Check if there are enough samples for modeling
    if len(state_data) < 2:
        print(f"Not enough data for state {state}. Skipping...")
        continue
    
    # Prepare data
    X = state_data.drop(['2023_yield', 'state', 'yield_group', 'commonland'], axis=1).copy()
    y = state_data['2023_yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    n_observations = len(y_pred)
#     print(f"\nState: {state}")
#     print("Mean Squared Error:", mse)
#     print("R^2 Score:", r2)
#     print("Number of predictions made:", n_observations)
    
    # Store results for plotting
    results.append((state, y_test, y_pred))
    
    # Store aggregated results
    agg_results.append({'state': state, 'r2_value': r2, 'mse': mse, 'rmse': rmse, 'n_observations': n_observations})

# Create DataFrame from aggregated results
agg_results_df = pd.DataFrame(agg_results)

print(agg_results_df)
agg_results_df.to_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_plots_v4_20250415_PDreal\RF_prediction_bystate_v4.csv")
# Plotting results
plt.figure(figsize=(10, 6))
for state, y_test, y_pred in results:
    plt.scatter(y_test, y_pred, alpha=0.3, label=state)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs. Predicted Yield by State')
plt.legend()
plt.show()


# # RF no BSI

# In[8]:


# no bsi index

# Assuming RF_data is defined and preprocessed
RF_data3 = RF_data1[RF_data1['max_index_red'] > 0]
RF_data3 = RF_data3.dropna()
RF_data3 = RF_data3.drop(columns='state')

# Remove columns that contain 'bsi' in their headers
RF_data3 = RF_data3[RF_data3.columns[~RF_data3.columns.str.contains('BSI')]]

# Prepare data
X = RF_data3.drop(['2023_yield', 'yield_group', 'commonland'], axis=1).copy()
y = RF_data3['2023_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)
print("Number of predictions made:", len(y_pred))

# Visualization of Actual vs Predicted Yield
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label= f'Root Mean Squared Error = {rmse}\nR^2 Score = {r2}\nNumber of predictions made = {len(y_pred)}')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs. Predicted Yield')
plt.legend()  # Don't forget to call plt.legend() to show the legend on the plot
plt.savefig(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_plots_v4_20250415_PDreal\RF_NO_BSI_prediction_v4.png")  # Specify the path and file name to save
plt.show()

# Feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting Feature Importances
plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='b', align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Relative Importance')
plt.legend()  # Don't forget to call plt.legend() to show the legend on the plot
plt.savefig(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_plots_v4_20250415_PDreal\RF_NO_BSI_featureImoprtance_v4.png")  # Specify the path and file name to save
plt.show()


# # Only NDMI + PRISM

# In[9]:


print(RF_data1)
# List of specific columns to keep
specific_columns = ['mean_elevation', 'pptMean', 'clucalcula', 'GDD', 'Adj_GDD', 'mean_slope', '2023_yield']

# Filter columns that have 'NDMI'RF_data1in their names
ndmi_columns = [col for col in RF_data1.columns if 'NDMI' in col or 'lai' in col]

# Combine the specific columns and NDMI columns
columns_to_keep = specific_columns + ndmi_columns

# Filter the dataframe
RF_data_new = RF_data1[columns_to_keep]

print(RF_data_new.info())



# # Generate new columns by multiplying each column with all other columns
# for i in range(len(numeric_columns)):
#     for j in range(i + 1, len(numeric_columns)):
#         new_column_name = f"{numeric_columns[i]}_x_{numeric_columns[j]}"
#         RF_data1[new_column_name] = RF_data1[numeric_columns[i]] * RF_data1[numeric_columns[j]]

# # Display the resulting dataframe
# print(RF_data1)


# In[10]:


columns = RF_data_new.columns
# Generate new columns by multiplying each column with all other columns
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        new_column_name = f"{columns[i]}_x_{columns[j]}"
        RF_data_new[new_column_name] = RF_data_new[columns[i]] * RF_data_new[columns[j]]

# Display the resulting dataframe
print(RF_data_new)


# In[11]:


print(RF_data_new.columns)
# Remove columns with 'yield_' in their header
RF_data_new = RF_data_new.loc[:, ~RF_data_new.columns.str.contains('2023_yield_')]
RF_data_new = RF_data_new.loc[:, ~RF_data_new.columns.str.contains('_2023_yield')]

RF_data_new.columns


# In[12]:


X = RF_data_new.drop(['2023_yield'], axis=1)
print(X.columns)
y = RF_data_new['2023_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)
print("Number of predictions made:", len(y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs. Predicted Yield')

# Add trendline
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r--")

plt.show()

# Feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting Feature Importances
plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='b')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Relative Importance')
plt.show()


# # Corn fields distance and CDL history

# In[13]:


import geopandas as gpd

# Load the GeoPackage file containing the polygons
file_path = r"C:\Users\User\Documents\Yields_project\Proag_Yields\proag_22_23_24_2023_yiled_data_1clu_1unit_no_dup_CORN_distance.gpkg"
gdf = gpd.read_file(file_path)

# Make sure the distance column exists
distance_column = 'distance'  # Replace with the actual name of your distance column

distance_map = dict(zip(gdf['CLUID'], gdf['distance']))


# # distance and history
# 

# In[14]:


# Load the GeoPackage file containing the polygons
polygons = r"C:\Users\User\Documents\Yields_project\Proag_Yields\proag_22_23_24_2023_yiled_data_1clu_1unit_no_dup_CORN.gpkg"
corn_gdf = gpd.read_file(polygons)

corn_gdf['distance'] = corn_gdf['CLUID'].map(distance_map)



# In[15]:


# Select columns that contain 'CDL' in their name
# Example dictionary mapping CDL values to crop names
cdl_to_crop_name = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",
    61: "Fallow/Idle Cropland",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Clouds/No Data",
    82: "Developed",
    83: "Water",
    87: "Wetlands",
    88: "Nonag/Undefined",
    92: "Aquaculture",
    111: "Open Water",
    112: "Perennial Ice/Snow",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    131: "Barren",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grassland/Pasture",
    190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    228: "Dbl Crop Triticale/Corn",
    229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht",
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley",
    234: "Dbl Crop Durum Wht/Sorghum",
    235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton",
    240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans",
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    248: "Eggplants",
    249: "Gourds",
    250: "Cranberries",
    254: "Dbl Crop Barley/Soybeans"
}


# In[16]:


corn_gdf.info()
print(corn_gdf.columns)
# Columns to replace values in
columns_to_replace = ['2019CDLmajority', '2020CDLmajority', '2021CDLmajority', '2022CDLmajority', '2023CDLmajority']

# Replace CDL values with crop names in the specified columns
for col in columns_to_replace:
    corn_gdf[col] = corn_gdf[col].replace(cdl_to_crop_name)

print(corn_gdf)


# In[17]:


# Sort the DataFrame by the distance column in descending order
corn_gdf = corn_gdf.sort_values(by=distance_column, ascending=False)

# Calculate the mean distance for the top 100 rows
top_100_mean_distance = corn_gdf.head(100)['unit_number_yield'].mean()

# Calculate the mean distance for the rest of the rows
rest_mean_distance = corn_gdf.tail(len(corn_gdf) - 100)['unit_number_yield'].mean()

# Print the results
print(f"Mean yield for the top 100 distance rows: {top_100_mean_distance}")
print(f"Mean yield for the rest of the rows: {rest_mean_distance}")

# Calculate the mean distance for the top 100 rows
top_50_mean_distance = corn_gdf.head(50)['unit_number_yield'].mean()

# Calculate the mean distance for the rest of the rows
rest_mean_distance = corn_gdf.tail(len(corn_gdf) - 50)['unit_number_yield'].mean()

# Print the results
print(f"Mean yield for the top 50 distance rows: {top_50_mean_distance}")
print(f"Mean yield for the rest of the rows: {rest_mean_distance}")

# Calculate the mean distance for the top 100 rows
top_20_mean_distance = corn_gdf.head(20)['unit_number_yield'].mean()

# Calculate the mean distance for the rest of the rows
rest_mean_distance = corn_gdf.tail(len(corn_gdf) - 20)['unit_number_yield'].mean()

# Print the results
print(f"Mean yield for the top 20 distance rows: {top_20_mean_distance}")
print(f"Mean yield for the rest of the rows: {rest_mean_distance}")


# In[18]:


# Select the top 20 rows
top_20 = corn_gdf.head(20)

# Initialize a list to store the crop names for each row
row_crop_names = []

# Extract crop names from the specified columns for each row
for idx, row in top_20.iterrows():
    CLUID = row['CLUID']

    crop_names = []
    for col in columns_to_replace:
        crop_names.append(row[col])
    row_crop_names.append((CLUID, crop_names))
# Print the crop names for each row
for CLUID, crops in row_crop_names:
    print(f"Crop names for row with index {CLUID}:")
    count_corn = crops.count('Corn')
    print(crops, count_corn)


# # indices by yield groups

# In[19]:


column_plots = RF_data
column_plots.info()
# Specify the range of columns to plot by index numbers
start_index = 5  # Third column (0-based index)
end_index = len(column_plots.columns) - 1  # One before the last column

# Select columns by index range
columns_to_plot = column_plots.columns[start_index:end_index]

# Ensure the output directory exists
output_folder = r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\group1_plots_v4_20250415_PDreal\group1_yield_group_plots_v4"
os.makedirs(output_folder, exist_ok=True)

# Create individual plots for each column
for column in columns_to_plot:
    plt.figure(figsize=(10, 5))
    for label, group in column_plots.groupby('yield_group'):
        group[column].plot(kind='hist', alpha=0.5, label=str(label), bins=30)
    plt.title(column)
    plt.legend(title='Yield Group')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    # Save each figure by the column name
    plt.savefig(f'{output_folder}/{column}.png')  # Filename based on column title
    plt.show()
    plt.close()  # Close the figure to free memory


# In[ ]:





# In[ ]:





# In[ ]:




