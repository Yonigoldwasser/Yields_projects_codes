#!/usr/bin/env python
# coding: utf-8

# In[4]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


# In[5]:


PWId_unit = gpd.read_file(r"F:\globalag\clus_sandwich_test\globalAg_clus_with_all_data.gpkg")
unit_map = dict(zip(PWId_unit['CLUID'], PWId_unit['PWId']))
PWId_unit


# In[21]:


Tx_area = gpd.read_file(r"F:\globalag\clus_sandwich_test\globalAg_clus_with_all_data_GEE_version_v2.shp")
Tx_area['Pwid_new'] = Tx_area['PWId'].astype(str).str.split('_').str[-1]
area_map = dict(zip(Tx_area['Pwid_new'], Tx_area['TX_area']))
area_map


# In[6]:


XL_PROD = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Years_comparison\globalAg\XL_PROD_PlanetWatchers_Extract.csv")
XL_PROD_cotton = XL_PROD[XL_PROD['CommodityName'] == 'COTTON                                  ']
XL_PROD_cotton['PWId'] = XL_PROD_cotton['CLUID'].map(unit_map).astype(str).str.strip()
XL_PROD_cotton['PWId'].replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)
print(XL_PROD_cotton)

# Ensure 'PWId' and 'Year' are strings for consistent merging
XL_PROD_cotton['ReinsuranceYear'] = XL_PROD_cotton['ReinsuranceYear'].astype(str).str.strip()

# Keep only the necessary columns.map(unit_map).astype(str).str.strip()
XL_PROD_cotton_filtered = XL_PROD_cotton[['PWId', 'ReinsuranceYear', 'PlantedDate']].copy()

# Convert to datetime
XL_PROD_cotton_filtered['PlantedDate'] = pd.to_datetime(XL_PROD_cotton_filtered['PlantedDate'], errors='coerce')
XL_PROD_cotton_filtered['Year'] = XL_PROD_cotton_filtered['ReinsuranceYear']
XL_PROD_cotton_filtered['PWId'] = XL_PROD_cotton_filtered['PWId'].str.replace('.0', '', regex=False).str.strip()

XL_PROD_cotton_filtered = XL_PROD_cotton_filtered[XL_PROD_cotton_filtered['PWId'].notna() & (XL_PROD_cotton_filtered['PWId'] != '')]


XL_PROD_cotton_filtered.sort_values('PWId')


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import os


# Load your data
df = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit_reshaped.csv")
data_for_irr = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit.csv")  # Update this path


output_dir = r"F:\globalag\clus_sandwich_test\singleCluUnit_scatter_plots_indices_yield\plantingDOY_byPractice_byArea_yields"
os.makedirs(output_dir, exist_ok=True)

df['PWId'] = df['PWId'].astype(str).str.strip()
df['Year'] = df['Year'].astype(str).str.strip()
XL_PROD_cotton_filtered['PWId'] = XL_PROD_cotton_filtered['PWId'].astype(str).str.strip()
XL_PROD_cotton_filtered['Year'] = XL_PROD_cotton_filtered['Year'].astype(str).str.strip()


df_merged = df.merge(XL_PROD_cotton_filtered, on=['PWId', 'Year'], how='left')
df_merged['DOY'] = df_merged['PlantedDate'].dt.dayofyear
df_merged = df_merged[df_merged['PWId'].notna() & (df_merged['PWId'] != '')]
df_merged = df_merged.sort_values('PWId')

data_for_irr['PWId'] = data_for_irr['PWId'].astype(str).str.strip()
# Create mapping
irr_map = dict(zip(data_for_irr['PWId'], data_for_irr['PracticeCode']))

# Apply mapping
df_merged['PracticeCode'] = df_merged['PWId'].map(irr_map)



# In[26]:


df_merged['PWId'] = df_merged['PWId'].astype(str).str.strip()
df_merged['TX_area'] = df_merged['PWId'].map(area_map)
df_merged.drop_duplicates(inplace=True)

print(df_merged)


# In[27]:


# df_merged['TX_area'] = df_merged['TX_area'].astype(str).str.strip()
# df_merged['PracticeCode'] = df_merged['PracticeCode'].astype(str).str.strip()

# Ensure the required columns are present
required_cols = ['TX_area', 'PracticeCode', 'DOY', 'yield']
missing_cols = [col for col in required_cols if col not in df_merged.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

r2_records = []

# Iterate over each TX_area and PracticeCode combination
for area in df_merged['TX_area'].unique():
    for practice in df_merged['PracticeCode'].unique():
        subset = df_merged[(df_merged['TX_area'] == area) & (df_merged['PracticeCode'] == practice)]
        
        # Skip empty subsets
        if subset.empty:
            continue
        
        X = subset[['DOY']]
        y = subset['yield']
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Record R² value
        r2_records.append({
            'TX_area': area,
            'PracticeCode': practice,
            'R2': r2
        })
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='DOY', y='yield', data=subset, alpha=0.7)
        plt.plot(subset['DOY'], y_pred, color='red', label=f"R² = {r2:.2f}")
        plt.title(f"Yield vs Planting DOY\nTX Area: {area}, Practice Code: {practice}")
        plt.xlabel("Planting DOY")
        plt.ylabel("Yield")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        filename = f"TX_area_{area}_PracticeCode_{practice}_AllYears_DOY_yield.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

# Create R² DataFrame
r2_df = pd.DataFrame(r2_records)
r2_df.to_csv(r"F:\globalag\clus_sandwich_test\singleCluUnit_scatter_plots_indices_yield\plantingDOY_byPractice_yields\plantingDOY_byPractice_ByArea_allYears_yields_r2.csv")


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import os

# --- Load Data ---
df = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit_reshaped.csv")
data_for_irr = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit.csv")  # Update this path

# Ensure both are string type and trimmed
df['PWId'] = df['PWId'].astype(str).str.strip()
data_for_irr['PWId'] = data_for_irr['PWId'].astype(str).str.strip()

# Create mapping for PracticeCode
irr_map = dict(zip(data_for_irr['PWId'], data_for_irr['PracticeCode']))
df['PracticeCode'] = df['PWId'].map(irr_map)

# Clean unwanted column if present
if '_yie_yield' in df.columns:
    df.drop(columns='_yie_yield', inplace=True)

# --- Setup ---
yield_col = 'yield'
practice_col = 'PracticeCode'
# output_dir = r"F:\globalag\clus_sandwich_test\singleCluUnit_scatter_plots_indices_yield\indices_yield_byPractice"
# os.makedirs(output_dir, exist_ok=True)

# Ensure 'yield' is numeric
assert yield_col in df.columns, f"'{yield_col}' not found in DataFrame"
assert pd.api.types.is_numeric_dtype(df[yield_col]), f"'{yield_col}' must be numeric"
assert practice_col in df.columns, f"'{practice_col}' not found in DataFrame"

# Numeric feature columns (excluding 'yield' and 'Year' if present)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in [yield_col, 'Year']]

# --- Process by PracticeCode ---
for practice, group_df in df.groupby(practice_col):
    r2_scores = {}

    # Convert 'Year' to string for consistent hue handling
    group_df['Year'] = group_df['Year'].astype(str)

    for col in feature_cols:
        # Drop rows with NaNs in the relevant columns
        sub_df = group_df[[col, yield_col, 'Year']].dropna()
        if sub_df.empty or sub_df[col].nunique() < 2:
            continue

        # Extract X (feature) and y (target)
        X = sub_df[[col]].values
        y = sub_df[yield_col].values

        # Fit linear regression
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        r2_scores[col] = r2

#         # Plot the scatter with trendline
#         plt.figure(figsize=(8, 6))
#         sns.scatterplot(data=sub_df, x=col, y=yield_col, hue='Year', alpha=0.8, palette='tab10', legend='brief')
#         plt.plot(sub_df[col], y_pred, color='red', linewidth=2, label='Regression Line')
#         plt.title(f'{col} vs {yield_col}\nPracticeCode: {practice} | $R^2$ = {r2:.3f}')
#         plt.xlabel(col)
#         plt.ylabel(yield_col)
#         plt.grid(True)
#         plt.legend(loc='upper right')
#         plt.tight_layout()
#         plt.show()
#         # Save plot
#         filename = f"{practice}_{col}_vs_{yield_col}.png".replace("/", "_")
#         plt.savefig(os.path.join(output_dir, filename))
#         plt.close()

    # Print top 5 R2 scores for this practice (optional)
    sorted_r2 = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 5 R2 scores for PracticeCode {practice}:")
    for feature, score in sorted_r2:
        print(f"{feature}: {score:.3f}")
sorted_r2


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import os

# --- Load Data ---
df = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit_reshaped.csv")
data_for_irr = pd.read_csv(r"F:\globalag\clus_sandwich_test\globalag_all_new_calc_indices_with_acreYield_thresholds_v2_singleCluUnit.csv")

# Ensure both are string type and trimmed
df['PWId'] = df['PWId'].astype(str).str.strip()
data_for_irr['PWId'] = data_for_irr['PWId'].astype(str).str.strip()

# Create mapping for PracticeCode
irr_map = dict(zip(data_for_irr['PWId'], data_for_irr['PracticeCode']))
df['PracticeCode'] = df['PWId'].map(irr_map)

# Clean unwanted column if present
if '_yie_yield' in df.columns:
    df.drop(columns='_yie_yield', inplace=True)

# --- Setup ---
yield_col = 'yield'
practice_col = 'PracticeCode'
year_col = 'Year'

# Ensure 'yield' is numeric
assert yield_col in df.columns, f"'{yield_col}' not found in DataFrame"
assert pd.api.types.is_numeric_dtype(df[yield_col]), f"'{yield_col}' must be numeric"
assert practice_col in df.columns, f"'{practice_col}' not found in DataFrame"
assert year_col in df.columns, f"'{year_col}' not found in DataFrame"

# Numeric feature columns (excluding 'yield' and 'Year' if present)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in [yield_col, 'Year']]

# --- Process by PracticeCode and Year ---
overall_r2 = []

for practice, group_df in df.groupby(practice_col):
    r2_scores = {}
    group_df[year_col] = group_df[year_col].astype(str)  # Ensure consistent year formatting

    # Calculate R² by practice
    for col in feature_cols:
        sub_df = group_df[[col, yield_col]].dropna()
        if sub_df.empty or sub_df[col].nunique() < 2:
            continue
        
        X = sub_df[[col]].values
        y = sub_df[yield_col].values

        # Fit linear regression
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        r2_scores[(practice, 'ALL', col)] = r2
        overall_r2.append((practice, 'ALL', col, r2))

    # Calculate R² by year within each practice
    for year, year_df in group_df.groupby(year_col):
        for col in feature_cols:
            sub_df = year_df[[col, yield_col]].dropna()
            if sub_df.empty or sub_df[col].nunique() < 2:
                continue
            
            X = sub_df[[col]].values
            y = sub_df[yield_col].values

            # Fit linear regression
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            r2_scores[(practice, year, col)] = r2
            overall_r2.append((practice, year, col, r2))

    # Print top 5 R² scores for this practice (optional)
    sorted_r2 = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 5 R2 scores for PracticeCode {practice} (Overall and by Year):")
    for (practice, year, feature), score in sorted_r2:
        print(f"{year} - {feature}: {score:.3f}")

# Convert to DataFrame for easier viewing
r2_df = pd.DataFrame(overall_r2, columns=['PracticeCode', 'Year', 'Feature', 'R2'])
r2_df.sort_values(by='R2', ascending=False, inplace=True)
print("\nTop 10 R² Scores Across All Practices and Years:")
print(r2_df.head(10))


# In[4]:


r2_df['R2'] = round(r2_df['R2'], 2)
r2_df


# In[ ]:




