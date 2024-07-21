import os
import pandas as pd

this_directory = os.path.dirname(__file__)

DATA_DIR = os.path.join(this_directory, '..', '..', 'data')

def process_data():
    with open(os.path.join(DATA_DIR, 'german_words_raw.txt'), 'r') as f:
        words = f.readlines()
    
    words = [word.strip() for word in words]
    words_df = pd.DataFrame(words, columns=['line'])
    words_df[["english", "singular", "plural"]] = words_df["line"].str.split("\t", expand=True)
    words_df["english"] = words_df["english"].str.split('. ').str[1]
    words_df[["article", "noun"]] = words_df["singular"].str.split(" ", n=1, expand=True)

    words_df["x"] = words_df["noun"].str.lower()

    words_df["y"] = words_df["article"].str.lower().map({"der": 0, "die": 1, "das": 2}).astype(int)

    print(words_df.groupby("article")["article"].count())
    print(words_df.head())

    words_df.to_csv(os.path.join(DATA_DIR, 'german_words_processed.txt'), index=False)