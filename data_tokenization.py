import csv
from collections import defaultdict

import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("porter_test")

stemmer = PorterStemmer()


class DataPoint:
    def __init__(self, features, label):
        self.features = features
        self.label = label

    def __repr__(self):
        return f"Features={self.features}, Label={self.label})"


def load_nrc_lexicon(filepath):
    emotion_lexicon = defaultdict(set)
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            word, emotion, association = line.strip().split("\t")
            if int(association) == 1:
                stemmed_word = stemmer.stem(word)
                emotion_lexicon[emotion].add(stemmed_word)
    return emotion_lexicon


def extract_features(text, emotion_lexicon):
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


def split_data(data):
    train_end = 30
    val_end = train_end + 9

    train_data = [data_pt for i, data_pt in enumerate(data) if i < train_end]
    val_data = [data_pt for i, data_pt in enumerate(data) if train_end <= i < val_end]
    test_data = [data_pt for i, data_pt in enumerate(data) if i >= val_end]

    return train_data, val_data, test_data


def read_and_process_file(nrc_filepath):
    emotion_lexicon = load_nrc_lexicon(nrc_filepath)
    data = []

    with open("cs173-hw3-processed.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            row_num, emotion, text = row
            features = extract_features(text, emotion_lexicon)
            data.append(DataPoint(features, emotion))

    return split_data(data)


if __name__ == "__main__":
    nrc_filepath = "data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    train_data, val_data, test_data = read_and_process_file(nrc_filepath)

    for i in range(10):
        print(train_data[i])
