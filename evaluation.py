import numpy as np
from data_tokenization import DataPoint, read_and_process_file
from logistic_regression import classifier
from sgd import best_lr_finder, sgd


def compute_confusion_matrix(
    test_data: list[DataPoint], w: list[float], b: float
) -> list[list[int]]:
    conf_matrix = np.zeros((2, 2), dtype=int)

    for data_pt in test_data:
        y_true = data_pt.label
        y_pred = int(classifier(data_pt, w, b) >= 0.5)
        conf_matrix[y_pred][y_true] += 1

    return conf_matrix


def print_confusion_matrix(conf_matrix):
    labels = ["Sadness", "Joy"]
    print("\n                ", end="")
    for label in labels:
        print(f"{label:<10}", end=" ")
    print()

    for i, row in enumerate(conf_matrix):
        print(f"{labels[i]:<10}", end="")
        print(" ".join(f"{x:10}" for x in row))


def compute_metrics(
    conf_matrix: list[list[int]], label: bool
) -> tuple[float, float, float, float]:
    if label:
        tp = conf_matrix[1][1]
        fn = conf_matrix[0][1]
        fp = conf_matrix[1][0]
        tn = conf_matrix[0][0]
    else:
        tp = conf_matrix[0][0]
        fn = conf_matrix[1][0]
        fp = conf_matrix[0][1]
        tn = conf_matrix[1][1]

    accuracy = 1.0 * (tp + tn) / conf_matrix.sum()
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    f1_score = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    train_data, val_data, test_data = read_and_process_file()

    learning_rate: float = best_lr_finder()[0]

    w, b = sgd(train_data, learning_rate=learning_rate)

    conf_matrix = compute_confusion_matrix(test_data, w, b)
    accuracy, precision, recall, f1_score = compute_metrics(conf_matrix, True)

    print("Confusion Matrix:")
    print_confusion_matrix(conf_matrix)
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
