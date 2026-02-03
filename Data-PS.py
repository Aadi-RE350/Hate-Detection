import pandas as pd
import re
import html
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import os

# 1. Convert to Pandas for efficient cleaning
# We access 'train' because load_dataset creates a 'train' split by default
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="Data/available_metahate.parquet"
)

df = dataset['train'].to_pandas()

# 2. Define the Cleaning Function
# This specifically targets the artifacts seen in your EDA (RTs, handles, HTML)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                   # Fix &amp; -> &
    text = re.sub(r'RT\s@\w+:', '', text)        # Remove "RT @user:"
    text = re.sub(r'@\w+', '', text)             # Remove remaining @mentions
    text = re.sub(r'http\S+', '', text)          # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra whitespace
    return text

# 3. Apply Cleaning
print(f"Cleaning {len(df)} rows...")
df['cleaned_text'] = df['text'].apply(clean_text)
# Remove any empty rows caused by cleaning
df = df[df['cleaned_text'].str.strip() != ""]

# 4. Stratified Split
# This ensures both Train and Test sets have exactly ~21% Hate labels
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],  # <--- CRITICAL for Imbalanced Data
    random_state=42
)

# 1. Define the folder paths
csv_folder = "csv"
data_folder = "Data"

# 2. Create directories if they don't exist
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

# 3. Save the full cleaned dataset (before split)
# This is useful if you want to change the split ratio later
df.to_csv(f"{csv_folder}/metahate_cleaned_full.csv", index=False)
df.to_csv(f"{data_folder}/metahate_cleaned_full.csv", index=False)

# 4. Save the Stratified Splits (Train and Test)
# This is crucial for reproducibility so your model always sees the same data
train_df.to_csv(f"{csv_folder}/train.csv", index=False)
train_df.to_csv(f"{data_folder}/train.csv", index=False)

test_df.to_csv(f"{csv_folder}/test.csv", index=False)
test_df.to_csv(f"{data_folder}/test.csv", index=False)

print(f"Files saved successfully in '{csv_folder}/' and '{data_folder}/'")
print(" - metahate_cleaned_full.csv")
print(" - train.csv")
print(" - test.csv")

print(f"Data ready. Train size: {len(train_df)}, Test size: {len(test_df)}")