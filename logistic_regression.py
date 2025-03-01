import numpy as np
from data_tokenization import DataPoint, read_and_process_file


def sigmoid(x: float) -> float:
    x = np.clip(x, -500, 500)
    return 1.0 / (1 + np.exp(-1.0 * x))


def classifier(data_point: DataPoint, w: list[float], b: float) -> float:
    x = np.dot(data_point.features, w) + b
    return sigmoid(x)


def binary_cross_entropy_loss(data_point: DataPoint, w: list[float], b: float) -> float:
    epsilon = 1e-9
    y_true = data_point.label
    y_pred = classifier(data_point, w, b)
    return -(
        y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
    )


if __name__ == "__main__":
    train_data = read_and_process_file()[0]
    example = [data_pt for data_pt in train_data if data_pt.row_num == 1]

    for data_pt in example:
        print(data_pt)
        print("LCE =", binary_cross_entropy_loss(data_pt, w=[0, 0, 0], b=0))
        print()
