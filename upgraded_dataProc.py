"""
dataProc.py — Data Preprocessing Pipeline
==========================================
This script handles label consolidation and initial data cleaning.
Sampling is performed INSIDE the main notebook AFTER the train/test split
to prevent data leakage.

Design Decisions:
  - Duplicates removed: duplicate rows cause data leakage and inflate metrics.
  - Inf/NaN removed: these break gradient-based optimisers.
  - Label consolidation: 40+ fine-grained labels → 7 semantic groups.
    Rationale: reduces label sparsity while preserving operationally
    meaningful distinctions (DDoS vs DoS vs Infiltration are very different
    threat types requiring different responses).
"""

import pandas as pd
import numpy as np

# --- Step 1: Load ---
df = pd.read_csv('Dataset/data.csv')
print(f"Raw shape: {df.shape}")

# --- Step 2: Remove Duplicates ---
# Exact duplicate rows offer no new information and can cause the same sample
# to appear in both train and test partitions (data leakage).
n_before = len(df)
df = df.drop_duplicates()
print(f"Removed {n_before - len(df)} duplicate rows.")

# --- Step 3: Remove NaN and Inf ---
# Infinite values appear in rate-based features (e.g. Flow Bytes/s) when
# flow duration is zero. These break StandardScaler and XGBoost's histogram builder.
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
n_before = len(df)
df = df.dropna()
print(f"Removed {n_before - len(df)} rows with NaN/Inf values.")

# --- Step 4: Label Consolidation ---
# Groups are defined by attack mechanism and operational severity:
#   Normal       — legitimate traffic
#   DDoS         — volumetric, distributed denial of service
#   DoS          — single-source denial of service
#   BruteForce   — credential attacks (SSH, FTP, web forms)
#   LowSeverity  — reconnaissance and web injection (lower immediate impact)
#   Botnet       — compromised host communication
#   Infiltration — stealthy lateral movement (highest severity)

label_mapping = {
    "Benign": "Normal", "BENIGN": "Normal",
    "DDoS-LOIC-HTTP": "DDoS", "DDoS-HOIC": "DDoS", "DDoS": "DDoS",
    "DDoS-NTP": "DDoS", "DDoS-TFTP": "DDoS", "DDoS-Syn": "DDoS",
    "DDoS-UDP": "DDoS", "DDoS-MSSQL": "DDoS", "DDoS-UDPLag": "DDoS",
    "DDoS-Ddossim": "DDoS", "DDoS-DNS": "DDoS", "DDoS-LDAP": "DDoS",
    "DDoS-SNMP": "DDoS", "DDoS-Slowloris": "DDoS", "DDoS-NetBIOS": "DDoS",
    "DoS-Hulk": "DoS", "DoS-Goldeneye": "DoS", "DoS-Slowloris": "DoS",
    "DoS-Slowhttptest": "DoS", "DoS-Slowread": "DoS", "DoS-Slowheaders": "DoS",
    "DoS-Rudy": "DoS", "DoS-Slowbody": "DoS", "DoS-Heartbleed": "DoS",
    "Bruteforce-SSH": "BruteForce", "Bruteforce-FTP": "BruteForce",
    "Webattack-bruteforce": "BruteForce",
    "Webattack-XSS": "LowSeverityAttack", "Webattack-SQLi": "LowSeverityAttack",
    "Portscan": "LowSeverityAttack",
    "Botnet": "Botnet",
    "Infiltration": "Infiltration"
}

df["Attack_Group"] = df["Label"].map(label_mapping)

# Check for any unmapped labels
unmapped = df[df["Attack_Group"].isna()]["Label"].unique()
if len(unmapped) > 0:
    print(f"WARNING — unmapped labels (will be dropped): {unmapped}")

# --- Step 5: Drop original label columns ---
# Now that Attack_Group is created, drop Label and ClassLabel
cols_to_drop = ["Label"]
if "ClassLabel" in df.columns:
    cols_to_drop.append("ClassLabel")

df = df.dropna(subset=["Attack_Group"]).drop(columns=cols_to_drop)

print("\nFinal class distribution:")
print(df['Attack_Group'].value_counts())
print(f"\nFinal cleaned shape: {df.shape}")

df.to_csv('Dataset/ProcessedData.csv', index=False)
print("\nSaved to Dataset/ProcessedData.csv")
print("\nNOTE: Sampling (SMOTETomek) is applied in the main notebook")
print("      AFTER train/val/test split to prevent data leakage.")