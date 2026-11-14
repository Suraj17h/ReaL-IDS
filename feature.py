import pandas as pd
import json

df = pd.read_csv('Dataset/ProcessedData.csv')

feature_cols = [
    'Total Fwd Packets', 'Fwd Packets Length Total', 'Fwd Packet Length Max',
    'Fwd Packet Length Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Bwd IAT Std', 'Fwd PSH Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Bwd Packets/s', 'Packet Length Mean', 'Packet Length Variance',
    'Init Fwd Win Bytes', 'Fwd Seg Size Min'
]

stats = {}
for cls in df['Attack_Group'].unique():
    subset = df[df['Attack_Group'] == cls][feature_cols]
    stats[cls] = subset.median().round(4).to_dict()

with open('class_feature_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Saved class_feature_stats.json")
print(json.dumps(stats, indent=2))