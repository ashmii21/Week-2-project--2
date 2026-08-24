import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("data/raw/dataset.xlsx")
df = df.dropna(how="all").dropna(axis=1, how="all")

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())

numeric = df.select_dtypes(include=np.number)
print("\nDescriptive statistics:\n", numeric.describe().T)
print("\nMean:\n", numeric.mean())
print("\nMedian:\n", numeric.median())
print("\nCount:\n", numeric.count())

# IQR outlier detection
for col in numeric.columns:
    s = numeric[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    print(f"\\n{col}: outliers = {((s < lower) | (s > upper)).sum()}")

# Numeric distributions
for col in numeric.columns:
    plt.figure(figsize=(7, 4))
    plt.hist(numeric[col].dropna(), bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
