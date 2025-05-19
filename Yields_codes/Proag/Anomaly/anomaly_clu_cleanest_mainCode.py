#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
import os
from datetime import datetime
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask

# Define paths and configurations
folder_path = r"F:\Anomaly_2025\yield_anomaly_proag2023_IL_500_clus_test\yield_anomaly_proag2023_500_clus_test"
output_dir = r"F:\Anomaly_2025\yield_anomaly_proag2023_IL_500_clus_test_results_raster"
poria_path = r"F:\Poria31032024Whole10m4326\Poria31032024Whole10m4326.tif" 
fields_vector_path = r"C:\Users\User\Documents\Yields_project\Anomaly_2025\IL_2024_sandwich_test\test_on_IL24_visit\IL24_visit_clusters.shp"

#### change for relevant years #########
first_year = 2019
last_year = 2023

date_format = "%Y%m%d"
satellite_priority = {'s2': 1, 'HLS2': 2, 'HLSL': 3, 'LS': 4}
polygon_column = 'id'
final_table_data = [] 
fields_vector = gpd.read_file(fields_vector_path)

# def count_pixels_in_polygon(folder_path, filename, fields_vector, polygon_name, polygon_column):
#     """
#     Counts the number of valid pixels inside a given polygon in a raster.
    
#     Parameters:
#         tiff_path (str): Path to the raster file (.tif).
#         vector_path (str): Path to the vector file (.shp).
#         polygon_name (str): The name or ID of the polygon to extract.
#         polygon_column (str): The column name containing polygon names/IDs.
    
#     Returns:
#         int: Number of valid pixels inside the polygon.
#     """
#     # Load raster
#     with rasterio.open(os.path.join(folder_path, filename)) as src:
#         raster_crs = src.crs  # Get CRS of raster

#     # Load vector
#     vector_data = fields_vector

#     # Ensure vector has the same CRS as raster
#     if vector_data.crs != raster_crs:
#         vector_data = vector_data.to_crs(raster_crs)

#     # Filter vector layer by polygon name
#     filtered_vector = vector_data[vector_data[polygon_column] == polygon_name]

#     # Ensure the vector contains at least one polygon
#     if filtered_vector.empty:
#         raise ValueError(f"No polygon found with name '{polygon_name}' in the vector file.")

#     # Clip raster to vector extent
#     with rasterio.open(os.path.join(folder_path, filename)) as src:
#         # Convert vector to GeoJSON format
#         geometry = [mapping(geom) for geom in filtered_vector.geometry]
        
#         # Clip raster using the polygon geometry
#         clipped_raster, clipped_transform = mask(src, geometry, crop=True)
        
#         # Read the first band (assuming single-band raster)
#         clipped_raster = clipped_raster[0]

#     # Count non-NaN (valid) pixels inside the polygon
#     valid_pixel_count = np.count_nonzero(~np.isnan(clipped_raster))

#     return valid_pixel_count

# Helper functions
def get_satellite_priority(satellite):
    return satellite_priority.get(satellite, float('inf'))

def extract_date_from_filename(band_name, satellite):
    if satellite in ['LS', 'HLS2', 'HLSL']:
        return datetime.strptime(band_name.split('_')[-4][:8], date_format)
    elif satellite == 's2':
        return datetime.strptime(band_name.split('_')[-5][:8], date_format)

def align_band_to_reference(data, src_transform, src_crs, reference_transform, reference_shape, reference_crs):
    aligned_data = np.zeros(reference_shape, dtype=rasterio.float32)
    reproject(
        source=data,
        destination=aligned_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=reference_transform,
        dst_crs=reference_crs,
        resampling=Resampling.bilinear
    )
    return aligned_data

def check_number_years_of_bands_in_alignedBands(bands):
    # Count the number of 2024 bands
    count_last_year = sum(1 for _, date, _, _, _ in bands if date.year == last_year)    
    total_count = len(bands)
#     print('count_last_year', count_last_year)
#     print('total_count', total_count)
    # Check if the current band's year is not 2024 or if there is only one band which is from 2024
    if count_last_year == 1 and total_count== 1 or (count_last_year == 0):
#         print('failed')
        failed = 'Missing history or current year images'
        confidence = ''
        final_table_data.append((polygon_name, confidence, failed))
        return False, total_count

    else:
#         print('continue')
        return True, total_count
            
def align_and_clip_raster(source_raster_path, target_meta, target_shape):
    """
    Reprojects and clips the source raster based on the target's metadata and shape.

    :param source_raster_path: Path to the source raster.
    :param target_meta: Metadata dictionary of the target raster.
    :param target_shape: Shape tuple (height, width) of the target raster.
    :return: A numpy array with the reprojected and clipped raster data.
    """
    with rasterio.open(source_raster_path) as src:
        # Calculate the transformation and new shape if not provided
        transform, width, height = calculate_default_transform(
            src.crs, target_meta['crs'], src.width, src.height, *src.bounds, 
            dst_width=target_shape[1], dst_height=target_shape[0]
        )
        # Destination array
        dest = np.zeros(target_shape, dtype=src.dtypes[0])

        # Perform the reprojection
        reproject(
            source=rasterio.band(src, 5),  # Assuming you want to reproject the fifth band
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_meta['transform'],
            dst_crs=target_meta['crs'],
            resampling=Resampling.nearest
        )
        
        return dest

# Process data
for filename in os.listdir(folder_path):#[2:4]:
    if filename.endswith(".tif"):
#         print("")
#         print(filename)
#         print("")
        all_bands = {}
        
        polygon_name = "_".join(filename.split("_")[:3])
        print(polygon_name)
        years = {str(year): None for year in range(first_year, last_year + 1)}


        # check for best bands of specific index for all years for creating history and diff rasters
        with rasterio.open(os.path.join(folder_path, filename)) as src:
            # Assume the first band is your reference
            reference_shape = (src.height, src.width)
            reference_transform = src.transform
            reference_crs = src.crs
            reference_meta = src.meta
            data = src.read()
            aligned_bands = []

            
            clu_total_pixel_count = np.count_nonzero(~np.isnan(data[-1]))
#             print('clu_total_pixel_count:', clu_total_pixel_count)
            
            for i in reversed(range(1, src.count + 1)):
                band_data = src.read(i, masked=True)
                band_name = src.descriptions[i - 1]
                year, satellite, index = band_name.split('_')[-3:]
                date = extract_date_from_filename(band_name, satellite)
                aligned_bands = []
                
                if index == 'lai':
                    valid_pixel_count = np.count_nonzero(~np.isnan(band_data.data))
                    
                    ###################### just for tile samples ###################################
#                     all_nan_pixels = np.all(np.isnan(data), axis=0)
#                     count_all_nan_pixels = np.sum(all_nan_pixels)
#                     raster_total_pixel_count = band_data.data.size
#                     clu_total_pixel_count = raster_total_pixel_count - count_all_nan_pixels
#                     print('clu_total_pixel_count2:', clu_total_pixel_count)

                 #####################################################################

#                     print('valid_pixel_count:', valid_pixel_count)
                    valid_pixel_percentage = round((valid_pixel_count / clu_total_pixel_count) * 100, 2)
#                     print("")
#                     print('raster_total_pixel_count:', raster_total_pixel_count)
#                     print('clu_total_pixel_count:', clu_total_pixel_count)
#                     print('valid_pixel_count:', valid_pixel_count)
#                     print('valid_pixel_percentage:', valid_pixel_percentage)
#                     print('unique_values ', np.unique(band_data))
                    satellite_rank = get_satellite_priority(satellite)

                    if year not in all_bands:
                        all_bands[year] = []

                    all_bands[year].append((date, satellite, valid_pixel_count, band_data.data.copy(), band_name, i, valid_pixel_percentage, src.transform, src.crs))
                    # [0] - date, [1]- satellite, [2] - valid_pixel_count, [3] - band_data.data.copy, [4] - band_name
                    # [5]- index, [6] - valid_pixel_percentage, [7]- src.transform, [8]- src.crs
        
#         for year, bands in all_bands.items():
#             for i in bands:
#                 print(i[4], i[2], i[1], i[5], i[6])

        # check best_band       
        for year, bands in all_bands.items():
#             print("")
#             print(year)
            best_band = None
            for band in bands:
                # check best_band sat priority
                if best_band is None or band[2] > best_band[2] or \
                   (band[2] == best_band[2] and get_satellite_priority(band[1]) < get_satellite_priority(best_band[1])):
                    best_band = band
#             print("")
#             print('best_band1:', best_band[4], best_band[1], best_band[2])
            
            
#             best_band2 = None
            best_date, best_pixels = best_band[0], best_band[2]
            for alt_band in bands:
                # Check if alt_band is later than best_band and meets other conditions
                if alt_band[0] > best_date and alt_band[6] >= 90:
                    best_band = alt_band
                    
#             print("")
#             print('best_band2:', best_band[4], best_band[1], best_band[2])
            
            ##Choose the best S2 band close in time and pixels count
            best_date2, best_pixels2 = best_band[0], best_band[2]

            for alt_band2 in bands:
                if alt_band2[1] == 's2':
                    # Calculate the difference in days between the best and alternative band
                    days_difference = (alt_band2[0] - best_date2).days

                    # Define the days difference condition based on whether the best_band is S2
                    if best_band[1] == 's2':
                        # If the best_band is S2, consider only positive days difference up to 20
                        days_condition = 1 <= days_difference <= 20
                    else:
                        # If the best_band is not S2, consider days difference within +/- 20 days
                        days_condition = -20 <= days_difference <= 20

                    # Check if the current band meets the days condition and pixel count threshold
                    if days_condition and alt_band2[6] >= 90:
                        best_band = alt_band2
             
 ################################### not ignoring less than 90% pixels images #######################################################                        
#             print('best_band:', best_band[4], best_band[1], best_band[2])
    
                        
#             print(f'Updated best band for year {year}:', best_band[4])

#             years[year] = best_band
#             aligned_bands.append((best_band[4], best_band[0], best_band[3], best_band[7], best_band[8]))
#         print('years', years)
#         for year, details in years.items():
#                 print("")
                
#                 print(f"{year}: {details[4]} (Band Index: {details[5]}, Pixels percentage: {details[6]})\n")
                
#         print('years', years)
#         print('aligned_bands_data', aligned_bands)
################################### ignore less than 90% pixels images ######################################################################
            if best_band[6] >= 90:
                years[year] = best_band
#                 print('best_band90', best_band)


                aligned_bands.append((best_band[4], best_band[0], best_band[3], best_band[7], best_band[8]))
#         print("")
#         print('yearsList', years)
#         print("")
#         print('best bands:')
#         for year, details in years.items():
#                 if details != None:
#                     print(f"{year}: {details[4]} (Band Index: {details[5]}, Pixels percentage: {details[6]})\n")
#         print('aligned_bands:')
        
        Bands_analyzed = []
        for i in aligned_bands:
            Bands_analyzed.append(i[0])
#             print('aligned_band', i[0])
        
        # checks if at least 1 year of history and that current year image exists, return total bands as well
        should_continue, total_bands_count = check_number_years_of_bands_in_alignedBands(aligned_bands)
        if not should_continue:
            continue  # Skip to the next iteration of the loop
###########################################################################################################################
    # Reproject all bands to the reference grid
        aligned_bands_data = [align_band_to_reference(data, transform, crs, reference_transform, reference_shape, reference_crs) 
                      for _, _, data, transform, crs in aligned_bands]  # Ignore best_band[4] and best_band[0] during the function call
#         print('aligned_bands_data', len(aligned_bands_data))
        
        # Calculations
        history_data_mean = np.nanmean(aligned_bands_data[1:], axis=0)  # Exclude the last year for historical average
        unique_values = np.unique(history_data_mean)
#         print("Number of unique values:", len(unique_values))
        fifth_year_data = aligned_bands_data[0]  # Last year data
        difference_data = fifth_year_data - history_data_mean
        history_band_amount = total_bands_count - 1
        difference_percentage_data = ((fifth_year_data - history_data_mean) / history_data_mean) * 100

        # cut poria for the clu raster size and mask diff data
        aligned_poria = align_and_clip_raster(poria_path, reference_meta, reference_shape)
        aligned_poria_mask = aligned_poria >= 1
        
        difference_data = np.where(aligned_poria_mask, difference_data, np.nan)
        difference_percentage_data = np.where(aligned_poria_mask, difference_percentage_data, np.nan)

        # Save the output
        output_meta = src.meta.copy()
        output_meta.update({
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw'
        })
#         print('filename', filename)
        
        history_path = os.path.join(output_dir, filename.split('.')[0] + '_history.tif')
        difference_path = os.path.join(output_dir, filename.split('.')[0] + '_difference.tif')
        difference_percentage_path = os.path.join(output_dir, filename.split('.')[0] + '_difference_percentage.tif')
        fifth_year_path = os.path.join(output_dir, filename.split('.')[0] + '_fifth_year.tif')
        
        z_score_path = os.path.join(output_dir, filename.split('.')[0] + '_Zscore.tif')
        diff_std_path = os.path.join(output_dir, filename.split('.')[0] + '_diff_std.tif')
        diff_percentile_path = os.path.join(output_dir, filename.split('.')[0] + '_diff_percentile.tif')

        
#         with rasterio.open(history_path, 'w', **output_meta) as dest:
#             dest.write(history_data_mean.astype(rasterio.float32), 1)

#         with rasterio.open(difference_path, 'w', **output_meta) as dest:
#             dest.write(difference_data.astype(rasterio.float32), 1)
        
#         with rasterio.open(difference_percentage_path, 'w', **output_meta) as dest:
#             dest.write(difference_percentage_data.astype(rasterio.float32), 1)

#         with rasterio.open(fifth_year_path, 'w', **output_meta) as dest:
#             dest.write(fifth_year_data.astype(rasterio.float32), 1)
        
#         if history_band_amount >= 2:
#             history_data_std = np.nanstd(aligned_bands_data[1:], axis=0)
#             z_score_data = np.where(history_data_std > 0, (fifth_year_data - history_data_mean) / history_data_std, 0)
# #             z_score_data = np.where(aligned_poria_mask, z_score_data, np.nan)
            
#             with rasterio.open(z_score_path, 'w', **output_meta) as dest:
#                 dest.write(z_score_data.astype(rasterio.float32), 1)
        
        ############################### raster stats fo df from diff_percenatge ###################################
        damaged_pixels = np.sum(difference_percentage_data < -40)
#         print('damaged_pixels', damaged_pixels)

        damaged_acres = damaged_pixels * 0.024710538
#         print('damaged_acres', damaged_acres)
        
        ############################### raster stats for df from diff std and percentile ###################################
#         # Calculate the mean and standard deviation
#         print('difference_data', difference_data)
#         # Find the unique numbers in the array
#         unique_numbers = np.unique(difference_data)

#         # Count of unique numbers
#         unique_count = len(difference_data)
#         print('unique_count', unique_count)
#         mean_value = np.nanmean(difference_data)
#         std_deviation = np.nanstd(difference_data)
#         print('mean_value', mean_value)
#         print('std_deviation', std_deviation)
#         # Define the threshold (e.g., 2 standard deviations below the mean)
#         threshold = mean_value - 4 * std_deviation
#         print('threshold', threshold)
#         # Create a mask for very low values
#         low_value_mask = difference_data < threshold
#         nodata_mask = difference_data == src.nodata
#         low_value_mask = np.less(difference_data, threshold, where=~np.isnan(difference_data))

#         # Count the number of low pixels
#         num_low_pixels = np.nansum(low_value_mask)
#         print("Number of low pixels:", num_low_pixels)

# #         with rasterio.open(diff_std_path, 'w', **output_meta) as dest:
# #             dest.write(low_value_mask.astype(rasterio.float32), 1)
       
        # Calculate the 5th percentile while ignoring NaNs
        percentile_5th = np.nanpercentile(difference_data, 5)

        # Create a mask for values below this percentile
        low_value_mask = np.less(difference_data, percentile_5th, where=~np.isnan(difference_data))

        # Count the number of low pixels (ignoring NaNs in the count)
        num_low_pixels = np.nansum(low_value_mask)
#         print("Number of low pixels percentile:", num_low_pixels)
        percentile_damaged_acres = num_low_pixels * 0.024710538
#         with rasterio.open(diff_percentile_path, 'w', **output_meta) as dest:
#             dest.write(low_value_mask.astype(rasterio.float32), 1)

                
##############################################################################
        #Export best band details to text file
        details_path = os.path.join(output_dir, filename.split('.')[0] + '_band_details.txt')
        with open(details_path, 'w') as f:
            
            confidence_medium = None
            for year, details in years.items():
                if details == None:
                    f.write(f"{year}: None\n")
                else:
                    f.write(f"{year}: {details[4]} (Band Index: {details[5]}, Pixels percentage: {details[6]})\n")
                    if 'LS' in details[4] or 'HLS' in details[4]:  
                        confidence_medium = 'yes'

            if confidence_medium == 'yes':
                confidence = 'Medium'
                failed = ''
                f.write(f"Confidence: {confidence}")
            else:
                confidence = 'High'
                failed = ''
                f.write(f"Confidence: {confidence}")

        # Append a tuple or list to the data list
        final_table_data.append((polygon_name, confidence, failed, history_band_amount, Bands_analyzed, damaged_acres, percentile_damaged_acres))

#         print(f"Files saved: {history_path}, {difference_path}, {fifth_year_path}")
df = pd.DataFrame(final_table_data, columns=['Filename', 'Confidence', 'Failed', 'History analyzed years', 'Bands analyzed', 'Damaged acres', 'Percentile damaged acres'])
df.to_csv(f'{output_dir}\proag2023_500_clus_anomaly_data.csv')
df


# In[ ]:


import pandas as pd
import geopandas as gpd

df1 = df
df2 = gpd.read_file(r"C:\Users\User\Documents\Yields_project\Anomaly_2025\IL_2024_sandwich_test\proag_2024_500_clus_test\proag_22_23_24_2023_yiled_data_1clu_1unit_no_dup_2023_CORN_yieldsHis.gpkg")

# Merge the DataFrames on the 'filename' column
merged_df = pd.merge(df1, df2, left_on='Filename', right_on='PWId')

# Calculate percent of damaged acres
merged_df['Field percent Damaged'] = round((merged_df['Damaged acres'] / merged_df['clucalcula']) * 100, 2)
merged_df['Field percent Damaged percentile'] = round((merged_df['Percentile damaged acres'] / merged_df['clucalcula']) * 100, 2)
merged_df['flag'] = merged_df['Field percent Damaged'] > 5
merged_df['flag percentile'] = merged_df['Field percent Damaged percentile'] > 5
merged_df.to_csv(f'{output_dir}\proag2023_500_clus_anomaly_data_flags.csv')

merged_df

