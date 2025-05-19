#!/usr/bin/env python
# coding: utf-8

# # scatter plot function

# In[43]:


######################################## plot function ###############
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

def plot_regression_scatter(data, x_column, y_column):
    x = data[x_column]
    y = data[y_column]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data points', alpha=0.3)
    plt.plot(x, intercept + slope * x, 'r', label=f'Fit line: $y={slope:.2f}x+{intercept:.2f}$')
    plt.title(f'Scatter Plot of {y_column} vs. {x_column} with Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.text(max(x), min(y), f'$R^2={r_value**2:.3f}$', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.show()
    print(f"R^2={r_value**2:.3f}")
    print(f"y={slope:.2f}x+{intercept:.2f}")
    print(f"Yield max {max(y):.2f}, min {min(y):.2f}")
    print(f"Index max {max(x):.2f}, min {min(x):.2f}")




# # 2022 Corn fields

# ## Schw84 2022

# In[29]:


import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\2022 Harvest.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\Schw84_harvest_grid_all_values_clean.shp")

points_shp = points_shp[(points_shp['Yield2022'] > 0) & (points_shp['Yield2022'] < 350)]
points_shp = points_shp[points_shp['Width'] >=600]


# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - this merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')

# Calculate the mean value of points in each polygon
# Assuming the value column in your points shapefile is named 'Value'
mean_values = joined_data.groupby('index_right')['Yield2022'].mean().reset_index()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')

# Print results and optionally save to new shapefile
print(polygons_with_means)
# polygons_with_means.to_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\Schw84_harvest_grid_all_values_clean_v2.shp")
schw84_2022 = polygons_with_means
LAI1 = '20220721_L'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield2022'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plot_regression_scatter(polygons_with_means, LAI1, 'Yield2022')


# ## GM100 2022

# In[30]:


import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\2022 Harvest cleaned.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\GM100_harvest_grid_all_values.shp")

points_shp = points_shp[(points_shp['Yield2022'] > 50) & (points_shp['Yield2022'] < 450)]
points_shp = points_shp[points_shp['Width'] >=600]


# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - this merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')

# Calculate the mean value of points in each polygon
# Assuming the value column in your points shapefile is named 'Value'
mean_values = joined_data.groupby('index_right')['Yield2022'].mean().reset_index()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')

# Print results and optionally save to new shapefile
print(polygons_with_means)
# polygons_with_means.to_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\GM100_harvest_grid_all_values_clean.shp")
gm100_2022 = polygons_with_means

LAI1 = '20220721_L'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield2022'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plot_regression_scatter(polygons_with_means, LAI1, 'Yield2022')


# ## Mccl70 2022

# In[44]:


# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\2022 Harvest cleaned.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid_all_values.shp")

points_shp = points_shp[(points_shp['Yield2022'] > 50) & (points_shp['Yield2022'] < 450)]
points_shp = points_shp[points_shp['Width'] >=600]

points_shp.info()
polygons_shp.info()

# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - this merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')
joined_data.info()
# Calculate the mean value of points in each polygon
# Assuming the value column in your points shapefile is named 'Value'
mean_values = joined_data.groupby('index_right')['Yield2022'].mean().reset_index()
mean_values.info()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')

# Print results and optionally save to new shapefile
print(polygons_with_means)
polygons_with_means.to_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid_all_values.shp")
mccl70_2022 = polygons_with_means

LAI1 = '20220721_L'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield2022'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plot_regression_scatter(polygons_with_means, LAI1, 'Yield2022')


# In[28]:


######################### Mccl70 2022 with 3 pixels boundry buffer ########################
# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\2022 Harvest cleaned.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid_all_values_Buffer_3_pixels.shp")

points_shp = points_shp[(points_shp['Yield2022'] > 50) & (points_shp['Yield2022'] < 450)]
points_shp = points_shp[points_shp['Width'] >=600]


# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - this merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')

# Calculate the mean value of points in each polygon
# Assuming the value column in your points shapefile is named 'Value'
mean_values = joined_data.groupby('index_right')['Yield2022'].mean().reset_index()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')

# Print results and optionally save to new shapefile
print(polygons_with_means)
LAI1 = '20220721_L'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield2022'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plot_regression_scatter(polygons_with_means, LAI1, 'Yield2022')


# In[53]:


################# all 2022 3 corn fields ###################################

####################### All 2020 3 fields data ######################
gm100_2022.info()
mccl70_2022.info()
schw84_2022.info()

concat_2022 = pd.concat([gm100_2022, schw84_2022, mccl70_2022])
concat_2022.info()

LAI1 = '20220721_L'

plot_regression_scatter(concat_2022, LAI1, 'Yield2022')


# # Corn 2020

# ## GM100 2020 corn

# In[56]:


############################## GM100 2020 corn #####################
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\2020 Harvest.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\harvest_grid_GM100.shp")
points_shp.info()
points_shp = points_shp[(points_shp['Yield'] != 0) & (points_shp['Yield'] < 300)]
points_shp = points_shp[points_shp['Width'] >=600]


# Calculate the 5th and 95th percentiles for Yield
percentile_5 = points_shp['Yield'].quantile(0.05)
percentile_95 = points_shp['Yield'].quantile(0.95)

# # Identify and list values to be erased
# values_to_erase = points_shp[(points_shp['Yield'] <= percentile_5) | (points_shp['Yield'] >= percentile_95)]
# deleted_values = values_to_erase['Yield'].tolist()

# # Remove the identified values from the DataFrame
# df_clean = points_shp[(points_shp['Yield'] > percentile_5) & (points_shp['Yield'] < percentile_95)]

df_clean = points_shp

# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')

# Calculate the mean values of Yield in each polygon
mean_values = joined_data.groupby('index_right')['Yield'].mean().reset_index()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')
GM100_2020 = polygons_with_means

# polygons_with_means.to_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\GM100_harvest_grid_all_values.shp")


LAI1 = '20200810_L'
LAI2 = '20200825_L'
NDVI1 = '20200810_N'
NDVI2 = '20200825_N'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield'], bins=30, color='blue', alpha=0.7)
plt.axvline(percentile_5, color='red', linestyle='dashed', linewidth=2)
plt.axvline(percentile_95, color='green', linestyle='dashed', linewidth=2)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(polygons_with_means[LAI2], bins=30, color='green', alpha=0.7)
plt.title('Histogram of LAI2')
plt.xlabel('LAI2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plot_regression_scatter(polygons_with_means, LAI1, 'Yield')
plot_regression_scatter(polygons_with_means, LAI2, 'Yield')
plot_regression_scatter(polygons_with_means, NDVI1, 'Yield')
plot_regression_scatter(polygons_with_means, NDVI2, 'Yield')


# In[186]:


from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

def plot_regression_sections1(data, x_column, y_column):
    # Sort data based on the x_column
    data_sorted = data.sort_values(by=x_column)
    
    # Divide data into three equal parts based on the x_column
    num_rows = len(data_sorted)
    part_size = num_rows // 3
    parts = [data_sorted[i * part_size:(i + 1) * part_size] for i in range(3)]

    # Plot scatter plot of all data points
    plt.figure(figsize=(10, 6))
    plt.scatter(data_sorted[x_column], data_sorted[y_column], color='blue', label='Data points')

    # Plot trendlines for each section
    for i, part in enumerate(parts):
        part_x = part[x_column]
        part_y = part[y_column]
        slope, intercept, r_value, p_value, std_err = linregress(part_x, part_y)
        trendline_label = f'Part {i+1}: $y={slope:.2f}x+{intercept:.2f}$, $R^2={r_value**2:.3f}$'
        plt.plot(part_x, intercept + slope * part_x, label=trendline_label)

    plt.title(f'Scatter Plot of {y_column} vs. {x_column} with Trendlines for 3 Sections')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

# Example usage
data = pd.read_excel(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\GM100_trendline_demo.xlsx")
# Specify columns for x and y variables
x_column = 'Yield_2020'
y_column = 'yield_own_eqmean'

plot_regression_sections1(data, x_column, y_column)




# ## Mccl70 2020 corn

# In[58]:


############################## Mccl70 2020 corn #####################
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\2020 Harvest.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid.shp")
points_shp.info()
points_shp = points_shp[(points_shp['Yield'] != 0) & (points_shp['Yield'] < 310)]
points_shp = points_shp[points_shp['Width'] >=600]


df_clean = points_shp

# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')

# Calculate the mean values of Yield in each polygon
mean_values = joined_data.groupby('index_right')['Yield'].mean().reset_index()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')
polygons_with_means.info()
Mccl70_2020 = polygons_with_means
# polygons_with_means.to_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid_all_values.shp")


LAI1 = '20200810_L'
LAI2 = '20200825_L'
NDVI1 = '20200810_N'
NDVI2 = '20200825_N'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield'], bins=30, color='blue', alpha=0.7)
plt.axvline(percentile_5, color='red', linestyle='dashed', linewidth=2)
plt.axvline(percentile_95, color='green', linestyle='dashed', linewidth=2)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(polygons_with_means[LAI2], bins=30, color='green', alpha=0.7)
plt.title('Histogram of LAI2')
plt.xlabel('LAI2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plot_regression_scatter(polygons_with_means, LAI1, 'Yield')
plot_regression_scatter(polygons_with_means, LAI2, 'Yield')
plot_regression_scatter(polygons_with_means, NDVI1, 'Yield')
plot_regression_scatter(polygons_with_means, NDVI2, 'Yield')



# ## Mccl70 2020 corn buffered 3 pixels

# In[201]:


############################## Mccl70 2020 corn buffered 3 pixels #####################
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load shapefiles
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid_all_values_Buffer_1_pixels.shp")
polygons_shp.info()


LAI1 = '20200810_L'
LAI2 = '20200825_L'
NDVI1 = '20200810_N'
NDVI2 = '20200825_N'

plot_regression_scatter(polygons_shp, LAI1, 'Yield_2020')
plot_regression_scatter(polygons_shp, LAI2, 'Yield_2020')
plot_regression_scatter(polygons_shp, NDVI1, 'Yield_2020')
plot_regression_scatter(polygons_shp, NDVI2, 'Yield_2020')



# ## Schw84 2020 corn

# In[59]:


############################## Schw84 2020 corn #####################
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load shapefiles
points_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\2020 Harvest.shp")
polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\Schw84_harvest_grid.shp")
# points_shp.info()
points_shp = points_shp[(points_shp['Yield'] != 0) & (points_shp['Yield'] < 400)]
points_shp = points_shp[points_shp['Width'] >=600]
points_shp = points_shp[points_shp['Distance'] < 50]
points_shp.info()



# Ensure both shapefiles use the same coordinate reference system
if points_shp.crs != polygons_shp.crs:
    points_shp = points_shp.to_crs(polygons_shp.crs)

# Spatial join - merges the point data into the polygon where each point is contained
joined_data = gpd.sjoin(points_shp, polygons_shp, how="inner", op='within')

# Calculate the mean values of Yield in each polygon
mean_values = joined_data.groupby('index_right')['Yield'].mean().reset_index()

# Optionally, merge these means back with your polygons data for further analysis or export
polygons_with_means = polygons_shp.merge(mean_values, left_index=True, right_on='index_right')
Schw84_2020 = polygons_with_means

# polygons_with_means.to_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\Schw84_harvest_grid_all_values.shp")


LAI1 = '20200810_L'
LAI2 = '20200825_L'
NDVI1 = '20200810_N'
NDVI2 = '20200825_N'

# Histograms for Yield, LAI1, and LAI2
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.hist(points_shp['Yield'], bins=30, color='blue', alpha=0.7)
plt.axvline(percentile_5, color='red', linestyle='dashed', linewidth=2)
plt.axvline(percentile_95, color='green', linestyle='dashed', linewidth=2)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(polygons_with_means[LAI1], bins=30, color='red', alpha=0.7)
plt.title('Histogram of LAI1')
plt.xlabel('LAI1')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(polygons_with_means[LAI2], bins=30, color='green', alpha=0.7)
plt.title('Histogram of LAI2')
plt.xlabel('LAI2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plot_regression_scatter(polygons_with_means, LAI1, 'Yield')
plot_regression_scatter(polygons_with_means, LAI2, 'Yield')
plot_regression_scatter(polygons_with_means, NDVI1, 'Yield')
plot_regression_scatter(polygons_with_means, NDVI2, 'Yield')



# ## All 2020 3 fields data

# In[66]:


####################### All 2020 3 fields data ######################
GM100_2020.info()
Mccl70_2020.info()
Schw84_2020.info()

concat_2020 = pd.concat([GM100_2020, Mccl70_2020, Schw84_2020])
concat_2020.info()

LAI1 = '20200810_L'
LAI2 = '20200825_L'
NDVI1 = '20200810_N'
NDVI2 = '20200825_N'

plot_regression_scatter(concat_2020, LAI1, 'Yield')
plot_regression_scatter(concat_2020, LAI2, 'Yield')
plot_regression_scatter(concat_2020, NDVI1, 'Yield')
plot_regression_scatter(concat_2020, NDVI2, 'Yield')


# # 2023 fields yields by 2020 3 fields equation

# In[212]:


####################### GL76 2023 real yield vs yields from 2020 3 fields equation ###################
import itertools
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_regression_scatter_shapefile(shapefile_path, x_column, y_column):
    # Read shapefile
    data = gpd.read_file(shapefile_path)
    data.info()
    data = data[(data[x_column] != 0) & (data[x_column] < 350)]
    data = data[(data[y_column] != 0) & (data[y_column] < 350)]
    # Extract x and y values
    x = data[x_column]
    y = data[y_column]

    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trendline = intercept + slope * x

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data points', alpha=0.3)
    plt.plot(x, trendline, 'r', label=f'Fit line: $y={slope:.2f}x+{intercept:.2f}$')
    plt.text(max(x), min(y), f'$R^2={r_value**2:.3f}$', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.title(f'Scatter Plot of {y_column} vs. {x_column} with Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

    # Print summary statistics
    print(f"Summary Statistics for {x_column}:")
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}, Std: {x.std()}")
    print(f"\nSummary Statistics for {y_column}:")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean()}, Std: {y.std()}")
    print(f"\nR^2={r_value**2:.3f}")
    print(f"y={slope:.2f}x+{intercept:.2f}")

# Example usage
shapefile_path = r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GL76 _ GL 76\GL76_harvest_grid_all_values.shp"  # Provide the path to your shapefile
x_column = 'Yield_2023'  # Name of the column representing the independent variable
y_column = '2023_from_'  # Name of the column representing the dependent variable

plot_regression_scatter_shapefile(shapefile_path, x_column, y_column)
###########################################################################################################################
def plot_regression_scatter_shapefile(shapefile_path, x_column, y_column):
    # Read shapefile
    data = gpd.read_file(shapefile_path)
    data.info()
    data = data[(data[x_column] > 100) & (data[x_column] < 350)]
    data = data[(data[x_column] > 100) & (data[y_column] < 350)]
    # Extract x and y values
    x = data[x_column]
    y = data[y_column]

    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trendline = intercept + slope * x

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data points', alpha=0.3)
    plt.plot(x, trendline, 'r', label=f'Fit line: $y={slope:.2f}x+{intercept:.2f}$')
    plt.text(max(x), min(y), f'$R^2={r_value**2:.3f}$', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.title(f'Scatter Plot of {y_column} vs. {x_column} with Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

    # Print summary statistics
    print(f"Summary Statistics for {x_column}:")
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}, Std: {x.std()}")
    print(f"\nSummary Statistics for {y_column}:")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean()}, Std: {y.std()}")
    print(f"\nR^2={r_value**2:.3f}")
    print(f"y={slope:.2f}x+{intercept:.2f}")

# Example usage
shapefile_path = r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GL76 _ GL 76\GL76_harvest_grid_all_values.shp"  # Provide the path to your shapefile
x_column = 'Yield_2023'  # Name of the column representing the independent variable
y_column = '2023_from_'  # Name of the column representing the dependent variable

plot_regression_scatter_shapefile(shapefile_path, x_column, y_column)


###########################################################################################################################
def plot_grouped_regression_scatter_shapefile(shapefile_path, x_column, y_column):
    # Read shapefile
    data = gpd.read_file(shapefile_path)
    data.info()
    
    # Remove outlier values
    data = data[(data[x_column] > 100) & (data[x_column] < 350)]
    data = data[(data[x_column] > 100) & (data[y_column] < 350)]
    
    # Define groups
    groups = {
        'Group 1 (0-200)': (data[x_column] <= 200),
        'Group 2 (200-300)': (data[x_column] > 200) & (data[x_column] <= 300),
        'Group 3 (>300)': (data[x_column] > 300)
    }
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    for group_name, group_condition in groups.items():
        group_data = data[group_condition]
        x = group_data[x_column]
        y = group_data[y_column]
        
        if not x.empty and not y.empty:
            # Calculate regression line for the group
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            trendline = intercept + slope * x
            
            # Plot
            plt.scatter(x, y, label=f'{group_name} Data Points', alpha=0.5)
            plt.plot(x, trendline, label=f'{group_name} Fit: $y={slope:.2f}x+{intercept:.2f}$, $R^2={r_value**2:.3f}$')
    
    plt.title('Grouped Scatter Plot with Regression Lines')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

# Example usage
shapefile_path = r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GL76 _ GL 76\GL76_harvest_grid_all_values.shp"  # Provide the path to your shapefile
x_column = 'Yield_2023'  # Independent variable
y_column = '2023_from_'  # Dependent variable

plot_grouped_regression_scatter_shapefile(shapefile_path, x_column, y_column)


############################################################################################################################
def best_grouped_regression_and_plot(shapefile_path, x_column, y_column):
    # Load shapefile
    data = gpd.read_file(shapefile_path)
    
    # Filter out zeros and obvious outliers
    data = data[(data[x_column] > 100) & (data[x_column] < 350)]
    data = data[(data[y_column] > 100) & (data[y_column] < 350)]

    # Generate all possible pairs of thresholds within the reasonable range
    max_value = data[x_column].max()
    min_value = data[x_column].min()
    thresholds = np.linspace(min_value, max_value, num=30)  # Adjust the number of splits as needed for performance vs. precision

    best_r2_sum = -np.inf
    best_thresholds = None
    best_group_data = None

    # Try all combinations of thresholds
    for threshold1, threshold2 in itertools.combinations(thresholds, 2):
        if threshold1 == threshold2:
            continue
        
        groups = {
            'Group 1 (< {:.2f})'.format(threshold1): data[data[x_column] <= threshold1],
            'Group 2 ({:.2f}-{:.2f})'.format(threshold1, threshold2): data[(data[x_column] > threshold1) & (data[x_column] <= threshold2)],
            'Group 3 (> {:.2f})'.format(threshold2): data[data[x_column] > threshold2]
        }

        r2_sum = 0
        group_data = {}
        for group_name, group in groups.items():
            if not group.empty:
                x = group[x_column]
                y = group[y_column]
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                r2 = r_value**2
                r2_sum += r2
                group_data[group_name] = (x, y, slope, intercept, r2)
        
        if r2_sum > best_r2_sum:
            best_r2_sum = r2_sum
            best_thresholds = (threshold1, threshold2)
            best_group_data = group_data

    # Plot the best combination
    plt.figure(figsize=(10, 6))
    for group_name, (x, y, slope, intercept, r2) in best_group_data.items():
        plt.scatter(x, y, label=f'{group_name} Data Points', alpha=0.5)
        trendline = slope * x + intercept
        plt.plot(x, trendline, label=f'{group_name}: $y={slope:.2f}x+{intercept:.2f}$, $R^2={r2:.3f}$')

    plt.title('Best Grouped Scatter Plot with Regression Lines')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

    # Output the best combination and its stats
    print(f"Best Thresholds: {best_thresholds}")
    print(f"Best Cumulative R²: {best_r2_sum}")
    for group_name, (x, y, slope, intercept, r2) in best_group_data.items():
        print(f"{group_name}: y = {slope:.2f}x + {intercept:.2f}, R² = {r2:.3f}")

# Example usage
shapefile_path = r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GL76 _ GL 76\GL76_harvest_grid_all_values.shp"  # Provide the path to your shapefile
x_column = 'Yield_2023'  # Independent variable
y_column = '2023_from_'  # Dependent variable

best_grouped_regression_and_plot(shapefile_path, x_column, y_column)


# In[235]:


####################### John87 2023 real yield vs yields from 2020 3 fields equation ###################
import itertools
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_regression_scatter_shapefile(shapefile_path, x_column, y_column):
    # Read shapefile
    data = gpd.read_file(shapefile_path)
    data.info()
    data = data[(data[x_column] > 0) & (data[x_column] < 350)]
    data = data[(data[y_column] > 0) & (data[y_column] < 350)]
    # Extract x and y values
    x = data[x_column]
    y = data[y_column]

    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trendline = intercept + slope * x

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data points', alpha=0.3)
    plt.plot(x, trendline, 'r', label=f'Fit line: $y={slope:.2f}x+{intercept:.2f}$')
    plt.text(max(x), min(y), f'$R^2={r_value**2:.3f}$', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.title(f'Scatter Plot of {y_column} vs. {x_column} with Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

    # Print summary statistics
    print(f"Summary Statistics for {x_column}:")
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}, Std: {x.std()}")
    print(f"\nSummary Statistics for {y_column}:")
    print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean()}, Std: {y.std()}")
    print(f"\nR^2={r_value**2:.3f}")
    print(f"y={slope:.2f}x+{intercept:.2f}")

# Example usage
shapefile_path = r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\John87 _ John 87\John 87_harvest_grid_all_values.shp"  # Provide the path to your shapefile
x_column = 'Yield_2023'  # Name of the column representing the independent variable
y_column = '2023by2022'  # Name of the column representing the dependent variable

plot_regression_scatter_shapefile(shapefile_path, x_column, y_column)


# # 2020 corn fields scatter plots by 2020 3 fields eq and 2020 own eq

# In[223]:


################# Schw84 2020 corn correlation to 2020 3 field eq or 2020 own eq ################################

polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Schw84 _ Schw 84\Schw84_harvest_grid_all_values.shp")
# points_shp.info()
new_points_shp = polygons_shp[(polygons_shp['Yield_2020'] > 0) & (polygons_shp['Yield_2020'] < 400)]
new_points_shp = new_points_shp[(new_points_shp[three_field_eq] > 0)]
new_points_shp = new_points_shp[(new_points_shp[own_eq] > 0)]
new_points_shp.info()

three_field_eq = '20_3field'
own_eq = '20_own'

plot_regression_scatter(new_points_shp, three_field_eq, 'Yield_2020')
plot_regression_scatter(new_points_shp, own_eq, 'Yield_2020')


# In[224]:


################# Mccl70 2020 corn correlation to 2020 3 field eq or 2020 own eq ################################

polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\Mccl70 _ Mccl 70\Mcll70_harvest_grid_all_values.shp")
# points_shp.info()
new_points_shp = polygons_shp[(polygons_shp['Yield_2020'] > 0) & (polygons_shp['Yield_2020'] < 400)]
new_points_shp = new_points_shp[(new_points_shp[three_field_eq] > 0)]
new_points_shp = new_points_shp[(new_points_shp[own_eq] > 0)]
new_points_shp.info()

three_field_eq = '20_3field'
own_eq = '20_own'

plot_regression_scatter(new_points_shp, three_field_eq, 'Yield_2020')
plot_regression_scatter(new_points_shp, own_eq, 'Yield_2020')


# In[225]:


################# GM 100 2020 correlation to 2020 3 field eq or 2020 own eq ################################

polygons_shp = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\GM100 _ GM 100\GM100_harvest_grid_all_values.shp")
# points_shp.info()
new_points_shp = polygons_shp[(polygons_shp['Yield_2020'] > 0) & (polygons_shp['Yield_2020'] < 400)]
new_points_shp = new_points_shp[(new_points_shp[three_field_eq] > 0)]
new_points_shp = new_points_shp[(new_points_shp[own_eq] > 0)]
new_points_shp.info()

three_field_eq = '20_3field'
own_eq = '20_own'

plot_regression_scatter(new_points_shp, three_field_eq, 'Yield_2020')
plot_regression_scatter(new_points_shp, own_eq, 'Yield_2020')


# # Random forest 2020 fields

# In[254]:


########## random forest for all LAI and NDVI values of 2020 fields and predict 2023 fields by their LAI and NDVI values #################

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load training data
train_data = pd.read_excel(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\2020_3fields_data.xlsx")
# Ensure the columns and target are correctly specified
target_train = target_train[(train_data['Yield_2020'] > 50) & (train_data['Yield_2020'] < 350)]
features_train  = train_data[['20200810_L']]#, '20200810_N']]#, '20200825_N', '20200825_L']]
target_train  = train_data['Yield_2020']

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features_train, target_train)

predict_data  = pd.read_excel(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\2023_2fields_data.xlsx")
predict_data['20200810_L'] = predict_data['20230731_L']
# predict_data['20200810_N'] = predict_data['20230731_N']
# predict_data['20200825_N'] = predict_data['20230914_N']
# predict_data['20200825_L'] = predict_data['20230914_L']

# Print feature importances
feature_importances = model.feature_importances_
feature_names = features_train.columns
importances = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
print("Feature Importances:")
print(importances.sort_values(by='Importance', ascending=False))

features_predict  = predict_data[['20200810_L']]#, '20200810_N']]#, '20200825_N', '20200825_L']]

# Predicting yield values for new data
predicted_yields = model.predict(features_predict)

# Optionally, save the predictions to a CSV file
predict_data['predicted_yield'] = predicted_yields


plot_regression_scatter(predict_data, 'predicted_yield', 'Yield_2023')



# ## R2 adjust

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load training data
df = pd.read_excel(r"C:\Users\User\Documents\Yields_project\Co-Alliance\Co-Alliance _ Lightfoot Luke-20240414T094834Z-001\Co-Alliance _ Lightfoot Luke\2020_3fields_data.xlsx")
# Ensure the columns and target are correctly specified
# df = df[(df['Yield_2020'] > 50) & (df['Yield_2020'] < 350)]

df = df.rename(columns=({"20200810_L": "LAI", "Yield_2020": "yield"}))
print(df)


# Assuming the DataFrame 'df' is already loaded and prepared

def adjust_r2(df, target_r2=0.7):
    current_r2 = 0
    removed_values = []

    x = df['LAI'].values.reshape(-1, 1)
    y = df['yield'].values
    indices = np.arange(len(df))

    while current_r2 < target_r2:
        current_r2 = r2_score(y, x)
        if current_r2 >= target_r2:
            break
        print("Lets go!")
        
        # Attempt to remove one point at a time and calculate the new R^2
        best_r2 = current_r2
        best_index = -1
        
        # Iterate over actual valid indices to find the best one to remove
        for i in range(len(x)):  # Ensuring that the loop runs within current array length
            # Try removing each point by skipping it in calculation
            temp_r2 = r2_score(np.delete(y, i), np.delete(x, i, axis=0))
            if temp_r2 > best_r2:
                best_r2 = temp_r2
                best_index = i

        if best_index != -1:
            # Record the value to be removed and update arrays
            removed_values.append(x[best_index][0])
            x = np.delete(x, best_index, axis=0)
            y = np.delete(y, best_index)
        else:
            # Exit if no improvement is found
            break
    
    # Return the adjusted DataFrame, removed values, and the final R^2
    return df.iloc[indices[:len(x)]], removed_values, current_r2

# Adjust the DataFrame to achieve the target R^2
adjusted_df, removed_LAI_values, final_r2 = adjust_r2(df)

# Plotting the adjusted data
plt.scatter(adjusted_df['LAI'], adjusted_df['yield'], label='Adjusted Data')
plt.xlabel('LAI')
plt.ylabel('Yield')
plt.title('Scatter Plot of LAI vs Yield')
plt.legend()
plt.show()

# Output the removed values and final R^2
print("Removed LAI values:", removed_LAI_values)
print("Final R^2 value:", final_r2)


# # 2020 and 2022 corn data

# In[83]:


# Renaming columns in the second DataFrame to match the first DataFrame

concat_2020['conYield'] = concat_2020['Yield']
concat_2022['conYield'] = concat_2022['Yield2022']

concat_2020['conLAI'] = concat_2020['20200810_L']
concat_2022['conLAI'] = concat_2022['20220721_L']
concat_2020_2022 = pd.concat([concat_2020, concat_2022])
concat_2020_2022.info()
concat_2020_2022 = concat_2020_2022[(concat_2020_2022['conYield'] > 0) & (concat_2020_2022['conYield'] < 350)]

plot_regression_scatter(concat_2020_2022, 'conYield', 'conLAI')


# In[ ]:




