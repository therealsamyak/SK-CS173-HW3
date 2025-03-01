from data_tokenization import read_and_process_file
from evaluation import compute_confusion_matrix, compute_metrics, print_confusion_matrix
from sgd import best_lr_finder, sgd

if __name__ == "__main__":
    # new features
    train_data_new, val_data_new, test_data_new = read_and_process_file(
        new_features=True
    )

    learning_rate_new, val_loss_new, _ = best_lr_finder(train_data_new, val_data_new)

    w_new, b_new = sgd(train_data_new, learning_rate_new)

    conf_matrix_new = compute_confusion_matrix(test_data_new, w_new, b_new)
    accuracy_new, precision_new, recall_new, f1_score_new = compute_metrics(
        conf_matrix_new, True
    )

    print("\n--- Results with New Features ---")
    print("Confusion Matrix:")
    print_confusion_matrix(conf_matrix_new)
    print()
    print(f"Validation Loss: {val_loss_new:.4f}")
    print(f"Accuracy: {accuracy_new:.4f}")
    print(f"Precision: {precision_new:.4f}")
    print(f"Recall: {recall_new:.4f}")
    print(f"F1 Score: {f1_score_new:.4f}")

    # old features
    train_data_old, val_data_old, test_data_old = read_and_process_file()

    learning_rate_old, val_loss_old, _ = best_lr_finder(train_data_old, val_data_old)

    w_old, b_old = sgd(train_data_old, learning_rate_old)

    conf_matrix_old = compute_confusion_matrix(test_data_old, w_old, b_old)
    accuracy_old, precision_old, recall_old, f1_score_old = compute_metrics(
        conf_matrix_old, True
    )

    print("\n--- Results with Old Features ---")
    print("Confusion Matrix:")
    print_confusion_matrix(conf_matrix_old)
    print()
    print(f"Validation Loss: {val_loss_old:.4f}")
    print(f"Accuracy: {accuracy_old:.4f}")
    print(f"Precision: {precision_old:.4f}")
    print(f"Recall: {recall_old:.4f}")
    print(f"F1 Score: {f1_score_old:.4f}")

    # comparison
    print("\n--- Performance Comparison ---")
    print(f"Validation Loss Change: {val_loss_new - val_loss_old:.4f}")
    print(f"Accuracy Change: {accuracy_new - accuracy_old:.4f}")
    print(f"Precision Change: {precision_new - precision_old:.4f}")
    print(f"Recall Change: {recall_new - recall_old:.4f}")
    print(f"F1 Score Change: {f1_score_new - f1_score_old:.4f}")
