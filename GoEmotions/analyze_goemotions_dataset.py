import pandas as pd
from glob import glob

# -----------------------------
# 1. LOAD DATASET
# -----------------------------

DATA_PATH = "./data/full_dataset"
csv_files = glob(f"{DATA_PATH}/goemotions_*.csv")

dfs = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dfs, ignore_index=True)

print("=" * 60)
print("GOEMOTIONS DATASET OVERVIEW (CLEAN)")
print("=" * 60)

print(f"\nTotal samples (rows): {len(df)}")
print(f"Total columns: {df.shape[1]}")

# -----------------------------
# 2. LOAD EMOTION LABEL LIST
# -----------------------------

with open("./data/emotions.txt", "r") as f:
    emotion_columns = [line.strip() for line in f.readlines()]

print(f"\nTotal emotion labels: {len(emotion_columns)}")
print("\nEmotion labels:")
print(emotion_columns)

# -----------------------------
# 3. EMOTION FREQUENCY
# -----------------------------

emotion_counts = df[emotion_columns].sum().sort_values(ascending=False)

print("\nEmotion frequency (top to bottom):")
print(emotion_counts)

# -----------------------------
# 4. MULTI-LABEL STATISTICS
# -----------------------------

df["num_emotions"] = df[emotion_columns].sum(axis=1)

print("\nMulti-label statistics:")
print(df["num_emotions"].describe())

print("\nSamples by number of emotions:")
print(df["num_emotions"].value_counts().sort_index())

# -----------------------------
# 5. NEUTRAL CHECK
# -----------------------------

if "neutral" in emotion_columns:
    print(f"\nNeutral-labeled samples: {df['neutral'].sum()}")

# -----------------------------
# 6. SAVE SUMMARY
# -----------------------------

emotion_counts.to_csv("emotion_frequency_summary_clean.csv")

print("\nSaved clean emotion frequency to: emotion_frequency_summary_clean.csv")
print("\n=== CLEAN ANALYSIS COMPLETE ===")
