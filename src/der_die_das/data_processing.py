import os

import pandas as pd
from sklearn.model_selection import train_test_split

from der_die_das.utils import DATA_DIR

this_directory = os.path.dirname(__file__)


def from_raw_to_processed() -> None:
    with open(os.path.join(DATA_DIR, "german_words_raw.txt")) as f:
        words = f.readlines()

    words = [word.strip() for word in words]
    words_df = pd.DataFrame(words, columns=["line"])
    words_df[["english", "singular", "plural"]] = words_df["line"].str.split("\t", expand=True)
    words_df.drop(columns=["line"], inplace=True)
    words_df["english"] = words_df["english"].str.split(". ").str[1]
    words_df[["article", "noun"]] = words_df["singular"].str.split(" ", n=1, expand=True)

    words_df["x"] = words_df["noun"].str.lower()

    words_df["y"] = words_df["article"].str.lower().map({"der": 0, "die": 1, "das": 2}).astype(int)

    words_df = words_df.drop_duplicates(subset=["x"])

    print(words_df.groupby("article")["article"].count())
    print(f"Number of words: {len(words_df)}")

    words_df.to_csv(os.path.join(DATA_DIR, "german_words_processed.csv"), index=False)


def split_train_and_test(language: str) -> None:
    if language == "german":
        words_df = pd.read_csv(os.path.join(DATA_DIR, "german_words_processed.csv"))
    elif language == "catalan":
        words_df = pd.read_csv(os.path.join(DATA_DIR, "catalan_words_processed.csv"))
    elif language == "croatian":
        words_df = pd.read_csv(os.path.join(DATA_DIR, "croatian_words_processed.csv"))

    words_df["length"] = words_df["x"].str.len()
    assert words_df[words_df["length"].isna()].empty

    words_df["length"] = words_df["length"].clip(lower=3, upper=13)
    if language == "croatian":
        words_df["length"] = words_df["length"].clip(lower=3, upper=10)

    print(words_df.groupby("length")["length"].count())

    words_df["stratify_label"] = words_df["y"].astype(str) + "_" + words_df["length"].astype(str)

    print(words_df.groupby("stratify_label")["stratify_label"].count())

    train_df, test_df = train_test_split(words_df, test_size=0.5, stratify=words_df["stratify_label"], random_state=42)

    train_df = train_df[["x", "y"]]
    test_df = test_df[["x", "y"]]

    if language == "german":
        train_name = "train.csv"
        test_name = "test.csv"
    elif language == "catalan":
        train_name = "train_cat.csv"
        test_name = "test_cat.csv"
    elif language == "croatian":
        train_name = "train_cro.csv"
        test_name = "test_cro.csv"

    train_df.to_csv(os.path.join(DATA_DIR, train_name), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, test_name), index=False)


def process_data(language: str, *, rerun_raw_to_processed: bool = False, rerun_train_test_split: bool = False) -> None:
    if language == "german":
        if not os.path.exists(os.path.join(DATA_DIR, "german_words_processed.csv")) or rerun_raw_to_processed:
            from_raw_to_processed()
        if not os.path.exists(os.path.join(DATA_DIR, "train.csv")) or rerun_train_test_split:
            split_train_and_test(language=language)
    elif language == "catalan":
        if not os.path.exists(os.path.join(DATA_DIR, "catalan_words_processed.csv")) or rerun_raw_to_processed:
            from_raw_to_processed_catalan()
        if not os.path.exists(os.path.join(DATA_DIR, "train_cat.csv")) or rerun_train_test_split:
            split_train_and_test(language=language)
    elif language == "croatian":
        if not os.path.exists(os.path.join(DATA_DIR, "croatian_words_processed.csv")) or rerun_raw_to_processed:
            from_raw_to_processed_croatian()
        if not os.path.exists(os.path.join(DATA_DIR, "train_cro.csv")) or rerun_train_test_split:
            split_train_and_test(language=language)


def translate_german_to_catalan() -> None:
    words_df = pd.read_csv(os.path.join(DATA_DIR, "german_words_processed.csv"))

    from deep_translator import MicrosoftTranslator

    translator = MicrosoftTranslator(
        source="de", target="ca", api_key="f8937130a4574616a8f61b0e50bf03a5", region="italynorth"
    )

    words_df["sing+plural"] = words_df["singular"] + ", " + words_df["plural"]
    words_df["catalan"] = translator.translate_batch(words_df["sing+plural"].values.tolist())

    words_df.to_csv(os.path.join(DATA_DIR, "catalan_words_raw.csv"), index=False)


def from_raw_to_processed_catalan() -> None:
    words_df = pd.read_csv(os.path.join(DATA_DIR, "catalan_words_raw.csv"))

    words_df[["singular", "plural"]] = words_df["catalan"].str.split(", ", expand=True)
    words_df["singular"] = words_df["singular"].str.lower().str.replace("l'", " ")
    words_df[["sing_article", "sing_noun"]] = words_df["singular"].str.split(" ", n=1, expand=True)
    words_df["plural"] = words_df["plural"].str.lower().str.replace("l'", " ")
    words_df[["plur_article", "plur_noun"]] = words_df["plural"].str.split(" ", n=1, expand=True)

    def get_article(row: pd.Series) -> int:
        assert row["sing_article"] in ["el", "la"] or row["plur_article"] in ["els", "les"]
        if row["sing_article"] == "el":
            return 0
        if row["sing_article"] == "la":
            return 1
        if row["plur_article"] == "els":
            return 0
        if row["plur_article"] == "les":
            return 1
        return -1

    words_df["y"] = words_df.apply(get_article, axis=1)
    words_df["x"] = words_df["sing_noun"]

    # remove " - " from the words
    words_df["x"] = words_df["x"].str.replace("-", "").str.strip()
    # if the word has a space, take the first word
    words_df["x"] = words_df["x"].str.split(" ").str[0]

    words_df = words_df[["english", "x", "y"]]

    # remove duplicates
    print(f"Number of words before removing duplicates: {len(words_df)}")
    words_df.drop_duplicates(subset=["x"], inplace=True)
    print(f"Number of words after removing duplicates: {len(words_df)}")

    words_df.to_csv(os.path.join(DATA_DIR, "catalan_words_processed.csv"), index=False)


def translate_german_to_croatian() -> None:
    words_df = pd.read_csv(os.path.join(DATA_DIR, "german_words_processed.csv"))

    from deep_translator import MicrosoftTranslator

    translator = MicrosoftTranslator(
        source="de", target="hr", api_key="f8937130a4574616a8f61b0e50bf03a5", region="italynorth"
    )

    words_df["possessive"] = words_df["singular"].apply(
        lambda x: x.replace("Der ", "Mein ").replace("Die ", "Meine ").replace("Das ", "Mein ")
    )
    words_df["croatian"] = translator.translate_batch(words_df["possessive"].values.tolist())

    words_df.to_csv(os.path.join(DATA_DIR, "croatian_words_raw.csv"), index=False)


def from_raw_to_processed_croatian() -> None:
    words_df = pd.read_csv(os.path.join(DATA_DIR, "croatian_words_raw.csv"))
    words_df["n_words"] = words_df["croatian"].str.split(" ").str.len()
    words_df = words_df[words_df["n_words"] == 2]  # noqa: PLR2004

    words_df[["possessive", "x"]] = words_df["croatian"].str.lower().str.split(" ", expand=True)
    words_df = words_df[words_df["possessive"].isin(["moj", "moja", "moje"])]

    def get_gender(possessive: str) -> int:
        if possessive == "moj":
            return 0
        if possessive == "moja":
            return 1
        if possessive == "moje":
            return 2
        return -1

    words_df["y"] = words_df["possessive"].apply(get_gender)
    assert set(words_df["y"]) == {0, 1, 2}

    # remove duplicates
    print(f"Number of words before removing duplicates: {len(words_df)}")
    words_df.drop_duplicates(subset=["x"], inplace=True)
    print(f"Number of words after removing duplicates: {len(words_df)}")

    words_df.to_csv(os.path.join(DATA_DIR, "croatian_words_processed.csv"), index=False)


if __name__ == "__main__":
    # translate_german_to_catalan()
    translate_german_to_croatian()
