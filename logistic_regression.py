import numpy as np
from data_tokenization import DataPoint, read_and_process_file


def sigmoid(x: float) -> float:
    return 1.0 / (1 + np.exp(-1.0 * x))


def classifier(
    data_point: DataPoint, w: list[float] = [0, 0, 0], b: float = 0
) -> float:
    x = np.dot(data_point.features, w) + b
    return sigmoid(x)


def binary_cross_entropy_loss(data_point: DataPoint) -> float:
    y_true = 1 if data_point.label == "Joy" else 0
    y_pred = classifier(data_point)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


if __name__ == "__main__":
    train_data = read_and_process_file()[0]
    example = [data_pt for data_pt in train_data if data_pt.row_num == 1]

    for data_pt in example:
        print(data_pt)
        print("LCE =", binary_cross_entropy_loss(data_pt))
        print()
