import pandas as pd
import re
from langdetect import detect, LangDetectException


def is_english(text):
    """
    Detect if text is in English.
    Returns True for English text, False otherwise.
    """
    # Check for Chinese, Japanese, Korean characters
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text):
        return False

    try:
        # Use langdetect for other languages
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If detection fails, assume it's not English
        return False


def split_parquet(
    input_path: str,
    train_size: int,
    test_size: int,
    output_prefix: str,
    random_state: int = 42,
):
    # Load the data
    df = pd.read_parquet(input_path)

    # Filter for English text only
    text_column = df.columns[0]  # Assuming the first column contains the text
    english_mask = df[text_column].apply(is_english)
    df_english = df[english_mask]

    print(f"Original data: {len(df)} rows")
    print(f"English-only data: {len(df_english)} rows")
    print(f"Removed {len(df) - len(df_english)} non-English rows")

    # Shuffle the filtered data
    df_shuffled = df_english.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    # Check if enough rows are available
    total_required = train_size + test_size
    if len(df_shuffled) < total_required:
        raise ValueError(
            f"Not enough rows: required {total_required}, but got {len(df_shuffled)}"
        )

    # Split
    train_df = df_shuffled.iloc[:train_size]
    test_df = df_shuffled.iloc[train_size : train_size + test_size]

    # Write to disk
    train_df.to_parquet(f"{output_prefix}_train.parquet", index=False)
    test_df.to_parquet(f"{output_prefix}_test.parquet", index=False)


if __name__ == "__main__":
    split_parquet(
        "./data/harmful.parquet",
        train_size=250,
        test_size=50,
        output_prefix="./data/harmful",
        random_state=42,
    )
    split_parquet(
        "./data/harmless.parquet",
        train_size=250,
        test_size=50,
        output_prefix="./data/harmless",
        random_state=43,
    )
