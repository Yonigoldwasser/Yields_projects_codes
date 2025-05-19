#!/usr/bin/env python
# coding: utf-8

# # high and medium fields prediction based on indices thresholds

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the CSV
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")

# Columns and thresholds to check
thresholds = {
    'max_index_lai': 5,
    'AUC_JUN_AUG_NDMI': 25,
    'AUC_JUN_AUG_MSAVI2': 45
}

# Create 'highYield' flag for each threshold column
for col, thresh in thresholds.items():
    df[f"{col}_high"] = df[col] > thresh

# Combine the flags into one
df['highYield'] = df[[f"{col}_high" for col in thresholds]].all(axis=1)

# Ground truth: 1 if yield group is High, else 0
df['actual_high'] = df['yield_group'] == 'High'

# Prediction
y_true = df['actual_high']
y_pred = df['highYield']

# Stats
total_high_yield = y_pred.sum()
true_positives = ((y_pred) & (y_true)).sum()
actual_high_total = y_true.sum()

# Print stats
print(f"Total 'highYield' predicted rows: {total_high_yield}")
print(f"Actual 'High' yield group rows: {actual_high_total}")
print(f"True Positives (correct 'highYield' predictions): {true_positives}")
print(f"Precision: {true_positives / total_high_yield:.2%} (of predicted highYield, how many are correct)")
print(f"Recall: {true_positives / actual_high_total:.2%} (of actual High group, how many were captured)")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low/Medium", "High"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - High Yield Prediction")
plt.show()
print('')

# Ground truth: 1 if yield group is High, else 0
df['actual_high'] = df['yield_group'].isin(['High', 'Medium'])

# Prediction
y_true = df['actual_high']
y_pred = df['highYield']

# Stats
total_high_yield = y_pred.sum()
true_positives = ((y_pred) & (y_true)).sum()
actual_high_total = y_true.sum()

# Print stats
print(f"Total 'highYield' predicted rows: {total_high_yield}")
print(f"Actual 'High' yield group rows: {actual_high_total}")
print(f"True Positives (correct 'highYield' predictions): {true_positives}")
print(f"Precision: {true_positives / total_high_yield:.2%} (of predicted highYield, how many are correct)")
print(f"Recall: {true_positives / actual_high_total:.2%} (of actual High group, how many were captured)")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low/Medium", "High"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - High Yield Prediction")
plt.show()
print('')




# # low fields prediction based on indices thresholds
# 

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the CSV
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")

# Columns and thresholds to check
thresholds = {
    'max_index_lai': 2.5,
    'AUC_JUN_AUG_NDMI': 10,
    'AUC_JUN_AUG_MSAVI2': 30
}

# Create 'lowYield' flag for each threshold column
for col, thresh in thresholds.items():
    df[f"{col}_low"] = df[col] < thresh

# Combine the flags into one
df['lowYield'] = df[[f"{col}_low" for col in thresholds]].all(axis=1)

# Ground truth: 1 if yield group is Low, else 0
df['actual_low'] = df['yield_group'] == 'Low'

# Prediction
y_true = df['actual_low']
y_pred = df['lowYield']

# Stats
total_low_yield = y_pred.sum()
true_positives = ((y_pred) & (y_true)).sum()
actual_low_total = y_true.sum()

# Print stats
print(f"Total 'lowYield' predicted rows: {total_low_yield}")
print(f"Actual 'low' yield group rows: {actual_low_total}")
print(f"True Positives (correct 'lowYield' predictions): {true_positives}")
print(f"Precision: {true_positives / total_low_yield:.2%} (of predicted lowYield, how many are correct)")
print(f"Recall: {true_positives / actual_low_total:.2%} (of actual low group, how many were captured)")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Medium/High", "Low"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - low Yield Prediction")
plt.show()
print('')





# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report
)

# Load the CSV
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")

# Thresholds for low prediction
low_thresholds = {
    'max_index_lai': 2.5,
    'AUC_JUN_AUG_NDMI': 10,
    'AUC_JUN_AUG_MSAVI2': 30
}

for col, thresh in low_thresholds.items():
    df[f"{col}_low"] = df[col] < thresh
df['lowYield'] = df[[f"{col}_low" for col in low_thresholds]].all(axis=1)

# Thresholds for high prediction
high_thresholds = {
    'max_index_lai': 5,
    'AUC_JUN_AUG_NDMI': 25,
    'AUC_JUN_AUG_MSAVI2': 45
}

for col, thresh in high_thresholds.items():
    df[f"{col}_high"] = df[col] > thresh
df['highYield'] = df[[f"{col}_high" for col in high_thresholds]].all(axis=1)

# Final prediction logic
def classify(row):
    if row['lowYield']:
        return 'Low'
    elif row['highYield']:
        return 'High'
    else:
        return 'Medium'

df['yield_predicted'] = df.apply(classify, axis=1)

# Actual vs predicted
y_true = df['yield_group']
y_pred = df['yield_predicted']

labels = ['Low', 'Medium', 'High']
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Multiclass Yield Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Totals
print("\n🔢 Total Actual Instances per Group:")
print(df['yield_group'].value_counts())

print("\n🔮 Total Predicted Instances per Group:")
print(df['yield_predicted'].value_counts())
# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Precision, Recall
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels, digits=2))


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report
)

# Load the CSV
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")

# Thresholds for low prediction
low_thresholds = {
    'max_index_lai': 2,
    'AUC_JUN_AUG_NDMI': 5,
    'AUC_JUN_AUG_MSAVI2': 35
}

for col, thresh in low_thresholds.items():
    df[f"{col}_low"] = df[col] < thresh
df['lowYield'] = df[[f"{col}_low" for col in low_thresholds]].all(axis=1)

# Thresholds for high prediction
high_thresholds = {
    'max_index_lai': 4,
    'AUC_JUN_AUG_NDMI': 20,
    'AUC_JUN_AUG_MSAVI2': 45
}

for col, thresh in high_thresholds.items():
    df[f"{col}_high"] = df[col] > thresh
df['highYield'] = df[[f"{col}_high" for col in high_thresholds]].all(axis=1)

# Final prediction logic
def classify(row):
    if row['lowYield']:
        return 'Low'
    elif row['highYield']:
        return 'High'
    else:
        return 'Medium'

df['yield_predicted'] = df.apply(classify, axis=1)

# Actual vs predicted
y_true = df['yield_group']
y_pred = df['yield_predicted']

labels = ['Low', 'Medium', 'High']
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Multiclass Yield Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Totals
print("\n🔢 Total Actual Instances per Group:")
print(df['yield_group'].value_counts())

print("\n🔮 Total Predicted Instances per Group:")
print(df['yield_predicted'].value_counts())
# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Precision, Recall
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=labels, digits=2))


# In[22]:


import pandas as pd
from sklearn.metrics import accuracy_score
from itertools import product

# Load data
df = pd.read_csv(r"C:\Users\User\Documents\Yields_project\Proag_Yields\Dan_data\all_indices_calc_v4_20250417_PDreal.csv")

# Define threshold ranges to test
lai_thresh_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 5.5, 6]
ndmi_thresh_range = [5, 10, 15, 20, 25, 30, 35]
msavi_thresh_range = [20, 25, 30, 35, 40, 45, 50, 55]

# Store best result
best_combo = None
best_accuracy = 0
print("started")
# Try all combinations
for low_lai, low_ndmi, low_msavi in product(lai_thresh_range, ndmi_thresh_range, msavi_thresh_range):
    for high_lai, high_ndmi, high_msavi in product(lai_thresh_range, ndmi_thresh_range, msavi_thresh_range):
        if (high_lai <= low_lai or high_ndmi <= low_ndmi or high_msavi <= low_msavi):
            continue  # Skip invalid ranges where high <= low

        # Create low yield flags
        df['lowYield'] = (df['max_index_lai'] < low_lai) & \
                         (df['AUC_JUN_AUG_NDMI'] < low_ndmi) & \
                         (df['AUC_JUN_AUG_MSAVI2'] < low_msavi)

        # Create high yield flags
        df['highYield'] = (df['max_index_lai'] > high_lai) & \
                          (df['AUC_JUN_AUG_NDMI'] > high_ndmi) & \
                          (df['AUC_JUN_AUG_MSAVI2'] > high_msavi)

        # Classify
        def classify(row):
            if row['lowYield']:
                return 'Low'
            elif row['highYield']:
                return 'High'
            else:
                return 'Medium'

        df['yield_predicted'] = df.apply(classify, axis=1)

        # Accuracy
        accuracy = accuracy_score(df['yield_group'], df['yield_predicted'])

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combo = {
                'low': {'lai': low_lai, 'ndmi': low_ndmi, 'msavi': low_msavi},
                'high': {'lai': high_lai, 'ndmi': high_ndmi, 'msavi': high_msavi},
                'accuracy': accuracy
            }

# Print best combination
print("\n🎯 Best Threshold Combination for Accuracy:")
print(f"Low thresholds => LAI < {best_combo['low']['lai']}, NDMI < {best_combo['low']['ndmi']}, MSAVI2 < {best_combo['low']['msavi']}")
print(f"High thresholds => LAI > {best_combo['high']['lai']}, NDMI > {best_combo['high']['ndmi']}, MSAVI2 > {best_combo['high']['msavi']}")
print(f"✅ Accuracy: {best_combo['accuracy']:.2%}")


# # RF all data

# In[12]:


import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import os

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.fillna(0)
X_test= X_test.fillna(0)

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


# In[7]:


# --- Top 10 Feature Importances ---
top_n = 10
top_indices = indices[:top_n]

plt.figure(figsize=(12, 6))
plt.title(f'Top {top_n} Feature Importances')
plt.bar(range(top_n), importances[top_indices], color='b')
plt.xticks(range(top_n), X_train.columns[top_indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.show()


# In[ ]:




