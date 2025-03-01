import csv
from collections import defaultdict

import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("porter_test")

stemmer = PorterStemmer()


class DataPoint:
    def __init__(self, features: list[float], label: str, row_num: int):
        self.features = features
        self.label = 1 if label == "Joy" else 0
        self.row_num = row_num

    def __repr__(self):
        return f"Row={self.row_num}, Label={self.label}, Features={self.features}"


def load_nrc_lexicon(filepath: str) -> dict[str, set[str]]:
    emotion_lexicon = defaultdict(set)
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            word, emotion, association = line.strip().split("\t")
            if int(association) == 1:
                stemmed_word = stemmer.stem(word)
                emotion_lexicon[emotion].add(stemmed_word)
    return emotion_lexicon


def extract_features(text: str, emotion_lexicon: dict[str, set[str]]) -> list[float]:
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    features = [
        # x1
        sum(1 for token in stemmed_tokens if token in emotion_lexicon["joy"]),
        # x2
        sum(1 for token in stemmed_tokens if token in emotion_lexicon["sadness"]),
        # x3
        len(stemmed_tokens),
    ]

    return features


def split_data(
    data: list[DataPoint],
) -> tuple[list[DataPoint], list[DataPoint], list[DataPoint]]:
    train_end = 30
    val_end = train_end + 9

    train_data = [data_pt for data_pt in data if data_pt.row_num < train_end]
    val_data = [data_pt for data_pt in data if train_end <= data_pt.row_num < val_end]
    test_data = [data_pt for data_pt in data if data_pt.row_num >= val_end]

    return train_data, val_data, test_data


def read_and_process_file() -> tuple[list[DataPoint], list[DataPoint], list[DataPoint]]:
    nrc_filepath = "data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    emotion_lexicon = load_nrc_lexicon(nrc_filepath)
    data = []

    with open("cs173-hw3-processed.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            row_num, emotion, text = row
            features = extract_features(text, emotion_lexicon)
            data.append(DataPoint(features, emotion, int(row_num)))

    return split_data(data)


if __name__ == "__main__":
    train_data, val_data, test_data = read_and_process_file()

    for i in range(10):
        print(train_data[i])
