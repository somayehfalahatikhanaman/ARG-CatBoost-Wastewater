"""
Load processed dataset for ARG prediction.
"""

import pandas as pd

df = pd.read_csv("data/ARG-Data-lastVersion.csv")

print(df.head())
