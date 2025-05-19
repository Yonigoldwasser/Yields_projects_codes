#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import rasterio
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Folder path containing the TIFF files
folder = r'F:\globalag\clus_sandwich_test\clus_sandwichs'

# Years of interest for analysis
years = ['2019', '2020', '2021', '2022']

# --- Function to Process a Single TIFF File ---
def process_tiff(raster_path):
    """Processes a single TIFF file to extract relevant data."""
    try:
        with rasterio.open(raster_path) as src:
            filename = os.path.basename(raster_path)
            PWId = filename.split('_')[2]
            results = {'PWId': PWId}
            original_band_names = src.descriptions
            if original_band_names is None:
                return None
            updated_band_names = ['_'.join(desc.split('_')[1:]) if desc else '' for desc in original_band_names]
            band_indices = {name: i + 1 for i, name in enumerate(updated_band_names)}
            print(updated_band_names[:20])
            # Identify total pixel count bands for each year
            total_pixel_bands = {name.split('_')[0]: band_indices[name] for name in updated_band_names if 'pixelCount' in name}
            print(f"total_pixel_bands: {total_pixel_bands}")

            # Calculate total pixel counts once per file
            total_pixel_counts = {}
            for year, band_index in total_pixel_bands.items():
                band_data = src.read(band_index)
                total_pixel_counts[year] = np.sum(band_data == 1)
            print(f"total_pixel_counts: {total_pixel_counts}")
            for band_name in updated_band_names:
                if 'pixelCount' in band_name or not band_name:
                    continue

                parts = band_name.split('_')
                if len(parts) < 2:
                    continue
                date_str = parts[0]
                index = parts[-1]
                year = date_str[:4]

                if year not in years:
                    continue

                band_data = src.read(band_indices[band_name])
                valid_values = band_data[~np.isnan(band_data)]
                clu_total_pixel_count = len(valid_values)

                total_pixels = total_pixel_counts.get(year)
                if total_pixels is None or total_pixels == 0:
                    continue

                pixel_percent = (clu_total_pixel_count / total_pixels) * 100
                if pixel_percent >= 80:
                    band_mean = np.mean(valid_values) if valid_values.size > 0 else np.nan
                    table_name = f'{date_str[:8]}_{index}'
                    results[table_name] = band_mean
            return results
    except rasterio.RasterioIOError as e:
        print(f"Error processing {os.path.basename(raster_path)}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {os.path.basename(raster_path)}: {e}")
        return None

# --- Process TIFF Files in Parallel ---
print("Starting processing of TIFF files...")
pw_results_list = []
# tiff_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]
tiff_files = [r"F:\globalag\clus_sandwich_test\clus_sandwichs_from_gee\globalag_sandwich_rasters\pa_unt_1191_TX_allYears.tif"]
print('tiff_files', tiff_files)
# Limit the number of files processed for testing (remove [:3] for full processing)
with ThreadPoolExecutor() as executor:
    results = executor.map(process_tiff, tiff_files)#[1:4])  # Process first 3 files for testing
    for result in results:
        if result:
            pw_results_list.append(result)

# Convert the list of results to a Pandas DataFrame
df_results = pd.DataFrame(pw_results_list)

print("\nFinal Results DataFrame:")
df_results


# In[ ]:


############################## fast version ##############################
import rasterio
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from tqdm import tqdm  # Progress bar

# Config
folder = r"F:\globalag\clus_sandwich_test\clus_sandwichs_from_gee\globalag_sandwich_rasters"
# NUM_FILES_TO_PROCESS = 30
NUM_WORKERS = multiprocessing.cpu_count()

def process_tiff_fast(raster_path):
    try:
        with rasterio.open(raster_path) as src:
            filename = os.path.basename(raster_path)
            pwid = filename.split('_')[2]
            band_names = src.descriptions
            band_indices = {name: i + 1 for i, name in enumerate(band_names)}

            total_pixel_counts = {}
            results = {'PWId': pwid}

            # Total pixel counts (pixelCount bands)
            for name, idx in band_indices.items():
                if 'pixelCount' in name:
                    year = name.split('_')[1]
                    band = src.read(idx)
                    total_pixel_counts[year] = np.sum(band == 1)

            # Data bands (choose best band per date/color)
            processed_bands = set()
            band_data_cache = {}

            for name, idx in band_indices.items():
                if 'pixelCount' in name or idx in processed_bands:
                    continue

                parts = name.split('_')
                if len(parts) < 6:
                    continue
                date_str = parts[1][:8]
                year = date_str[:4]
                color = parts[-1]
                simplified_name = f"{date_str}_{color}"

                # Find best band for this date/color
                candidate_indices = [
                    index for band_name, index in band_indices.items()
                    if len(band_name.split('_')) >= 6
                    and band_name.split('_')[1][:8] == date_str
                    and band_name.split('_')[-1] == color
                    and 'pixelCount' not in band_name
                ]

                max_valid_pixels = -1
                best_idx = None
                for candidate_idx in candidate_indices:
                    if candidate_idx not in band_data_cache:
                        band_data_cache[candidate_idx] = src.read(candidate_idx)
                    band_data = band_data_cache[candidate_idx]
                    valid_pixel_count = np.sum(~np.isnan(band_data))
                    if valid_pixel_count > max_valid_pixels:
                        max_valid_pixels = valid_pixel_count
                        best_idx = candidate_idx

                # Compute mean if valid percentage >= 80%
                if best_idx:
                    best_band_data = band_data_cache[best_idx]
                    valid_pixels = best_band_data[~np.isnan(best_band_data)]
                    total_pixels = total_pixel_counts.get(year)
                    if total_pixels and len(valid_pixels) / total_pixels * 100 >= 80:
                        mean_value = np.nanmean(valid_pixels) if valid_pixels.size > 0 else np.nan
                        results[simplified_name] = mean_value
                        processed_bands.add(best_idx)

            return results
    except Exception as e:
        print(f"Error processing {os.path.basename(raster_path)}: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting fast raster processing with ProcessPool ({NUM_WORKERS} workers)...")

    tiff_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = []
        for i, result in enumerate(executor.map(process_tiff_fast, tiff_files), 1):
            if result:
                results.append(result)
            if i % 100 == 0:
                print(f"Processed {i} TIFF files...")

    df_results = pd.DataFrame(results)
    print("\nFinal Results DataFrame:")
    print(df_results)
#     df_results.to_csv(r"F:\globalag\clus_sandwich_test\globalag_all_indices_data.csv", index=False)

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")


# In[ ]:




