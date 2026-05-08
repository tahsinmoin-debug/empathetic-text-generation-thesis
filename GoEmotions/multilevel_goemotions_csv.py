import pandas as pd
from glob import glob
from tqdm import tqdm

# -----------------------------
# 1. LOAD ALL CSV FILES
# -----------------------------

DATA_PATH = "./data/full_dataset"
csv_files = glob(f"{DATA_PATH}/goemotions_*.csv")

dfs = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dfs, ignore_index=True)

print(f"Total samples loaded: {len(df)}")

# -----------------------------
# 1.5 COLLAPSE DUPLICATE TEXTS (MERGE RATERS)
# -----------------------------

print("\nCollapsing duplicate texts (multi-rater issue)...")

# Identify emotion columns properly (binary 0/1)
emotion_columns = [
    col for col in df.columns
    if col not in ["text", "id", "author", "subreddit", "link_id",
                   "parent_id", "created_utc", "rater_id", "example_very_unclear"]
]

# Group by text and OR the emotion labels
df = (
    df.groupby("text", as_index=False)[emotion_columns]
      .max()
)

print(f"Samples after collapsing: {len(df)}")

# -----------------------------
# 2. DEFINE EMOTION GROUPS
# -----------------------------

POSITIVE_EMOTIONS = {
    "joy", "amusement", "excitement", "love",
    "gratitude", "pride", "optimism", "relief"
}

NEUTRAL_EMOTION = {"neutral"}

NEGATIVE_EMOTIONS = {
    "sadness", "grief", "disappointment", "remorse",
    "fear", "anxiety", "nervousness",
    "shame", "guilt", "embarrassment",
    "anger", "annoyance",
    "disgust",
    "jealousy", "envy"
}

# -----------------------------
# 3. LEVEL-3 → LEVEL-2 MAPPING
# -----------------------------

LEVEL2_MAPPING = {
    # Loss / Sadness
    "sadness": "Loss",
    "grief": "Loss",
    "disappointment": "Loss",
    "remorse": "Loss",

    # Fear / Anxiety
    "fear": "Fear",
    "anxiety": "Fear",
    "nervousness": "Fear",

    # Self-conscious
    "shame": "Self-Conscious",
    "guilt": "Self-Conscious",
    "embarrassment": "Self-Conscious",

    # Anger
    "anger": "Anger",
    "annoyance": "Anger",

    # Disgust
    "disgust": "Disgust",

    # Social pain
    "jealousy": "Social-Pain",
    "envy": "Social-Pain"
}

LEVEL2_PRIORITY = [
    "Loss",
    "Fear",
    "Self-Conscious",
    "Anger",
    "Disgust",
    "Social-Pain"
]

# -----------------------------
# 5. MULTI-LEVEL CLASSIFICATION
# -----------------------------

processed_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    labels = {emo for emo in emotion_columns if row[emo] == 1}

    negative_labels = labels & NEGATIVE_EMOTIONS

    # ❌ Remove positive-only or neutral-only samples
    if not negative_labels:
        continue

    # LEVEL-3: original negative emotions
    level3 = list(negative_labels)

    # LEVEL-2 mapping
    mapped_level2 = {
        LEVEL2_MAPPING[emo]
        for emo in level3
        if emo in LEVEL2_MAPPING
    }

    if not mapped_level2:
        continue

    # Choose dominant Level-2
    level2_final = None
    for cat in LEVEL2_PRIORITY:
        if cat in mapped_level2:
            level2_final = cat
            break

    processed_rows.append({
        "text": row["text"],
        "level_1": "Negative",
        "level_2": level2_final,
        "level_3": ",".join(level3)
    })

# -----------------------------
# 6. SAVE FINAL DATASET
# -----------------------------

final_df = pd.DataFrame(processed_rows)
final_df.to_csv("goemotions_multilevel_negative.csv", index=False)

print("\n=== DONE ===")
print(f"Final samples: {len(final_df)}")
print("\nLevel-2 distribution:")
print(final_df["level_2"].value_counts())
