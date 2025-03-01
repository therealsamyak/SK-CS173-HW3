import numpy as np
from data_tokenization import DataPoint, read_and_process_file
from logistic_regression import binary_cross_entropy_loss, classifier


def sgd(train_data: list[DataPoint], learning_rate: float, epochs: int = 100):
    w = np.zeros(len(train_data[0].features))
    b = 0

    for _ in range(epochs):
        np.random.shuffle(train_data)
        for data_point in train_data:
            y_true = data_point.label
            y_pred = classifier(data_point, w=w, b=b)

            error = y_pred - y_true
            w -= learning_rate * error * np.array(data_point.features)
            b -= learning_rate * error

    return w, b


def best_lr_finder(
    train_data: list[DataPoint],
    val_data: list[DataPoint],
    learning_rates: list[float] = [
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
    ],
):
    best_lr = None
    lowest_val_loss = float("inf")
    results = {}

    for lr in learning_rates:
        w, b = sgd(train_data, lr)
        val_loss = np.mean(
            [binary_cross_entropy_loss(data_pt, w, b) for data_pt in val_data]
        )
        results[lr] = val_loss

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_lr = lr

    return best_lr, lowest_val_loss, results


if __name__ == "__main__":

    train_data, val_data, _ = read_and_process_file()
    best_lr, lowest_val_loss, results = best_lr_finder(train_data, val_data)

    print()
    print(f"Best Learning Rate: {best_lr}")
    print(f"Lowest Validation Loss: {lowest_val_loss:.4f}")
    print()
    print("Validation Loss for Each Learning Rate:")
    for lr, loss in results.items():
        print(f"Learning Rate: {lr}, Loss: {loss:.4f}")
