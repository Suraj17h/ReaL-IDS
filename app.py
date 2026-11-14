import pandas as pd
df = pd.read_csv('Dataset/data.csv', nrows=5)  # reads just 5 rows
print(df.columns.tolist())
print(df.shape)
print(df['Label'].value_counts())