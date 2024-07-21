import os
import pandas as pd
from sklearn.model_selection import train_test_split


this_directory = os.path.dirname(__file__)

DATA_DIR = os.path.join(this_directory, '..', '..', 'data')

def from_raw_to_processed():
    with open(os.path.join(DATA_DIR, 'german_words_raw.txt'), 'r') as f:
        words = f.readlines()
    
    words = [word.strip() for word in words]
    words_df = pd.DataFrame(words, columns=['line'])
    words_df[["english", "singular", "plural"]] = words_df["line"].str.split("\t", expand=True)
    words_df.drop(columns=["line"], inplace=True)
    words_df["english"] = words_df["english"].str.split('. ').str[1]
    words_df[["article", "noun"]] = words_df["singular"].str.split(" ", n=1, expand=True)

    words_df["x"] = words_df["noun"].str.lower()

    words_df["y"] = words_df["article"].str.lower().map({"der": 0, "die": 1, "das": 2}).astype(int)

    print(words_df.groupby("article")["article"].count())
    print(words_df.head())

    words_df.to_csv(os.path.join(DATA_DIR, 'german_words_processed.csv'), index=False)


def split_tran_and_test():
    words_df = pd.read_csv(os.path.join(DATA_DIR, 'german_words_processed.csv'))

    words_df["length"] = words_df["x"].str.len()

    words_df["length"] = words_df["length"].clip(lower=3, upper=13)

    print(words_df.groupby("length")["length"].count())

    words_df['stratify_label'] = words_df['y'].astype(str) + "_" + words_df['length'].astype(str)

    print(words_df.groupby("stratify_label")["stratify_label"].count())

    train_df, test_df = train_test_split(words_df, test_size=0.5, stratify=words_df['stratify_label'])

    train_df = train_df[['x', 'y']]
    test_df = test_df[['x', 'y']]

    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)


def process_data(rerun_raw_to_processed: bool = False):
    if not os.path.exists(os.path.join(DATA_DIR, 'german_words_processed.csv')) or rerun_raw_to_processed:
        from_raw_to_processed()
    
    split_tran_and_test()
    
