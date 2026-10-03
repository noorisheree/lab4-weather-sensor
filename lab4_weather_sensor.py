import pandas as pd
import numpy as np
from scipy import stats

# Weather sensor temperature readings
temperatures = pd.Series([
    22, 23, np.nan, np.nan, np.nan,
    25, 26, 58, 24, 23,
    22, 21, 23, 24, 25
])

# Statistics before cleaning
print("Statistics Before Cleaning:")
print(temperatures.describe())

print("\nMissing Values Before Cleaning:", temperatures.isnull().sum())

# Linear interpolation for missing values
interpolated = temperatures.interpolate(method="linear")

print("\nAfter Linear Interpolation:")
print(interpolated)

# Calculate Z-scores
z_scores = stats.zscore(interpolated)

# Detect outliers using Z-score threshold of 2
outlier = abs(z_scores) > 2

print("\nOutlier Detection:")
print(outlier)

# Replace only flagged outliers with median
median_temp = interpolated.median()
cleaned = interpolated.copy()
cleaned[outlier] = median_temp

print("\nMedian Temperature:", median_temp)

print("\nCleaned Temperatures:")
print(cleaned)

# Standardize final cleaned temperature
standardized = (cleaned - cleaned.mean()) / cleaned.std()

# Before and after comparison
comparison = pd.DataFrame({
    "Before": temperatures,
    "After": cleaned,
    "Standardized": standardized
})

print("\nBefore and After Comparison:")
print(comparison)