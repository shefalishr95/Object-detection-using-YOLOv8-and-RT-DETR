from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, val_data):
    """Evaluate the model on validation data and return metrics."""
    preds = model.predict(val_data["images"])
    y_true = val_data["labels"]

    # Calculate metrics like precision, recall, etc.
    precision = precision_score(y_true, preds, average="weighted")
    recall = recall_score(y_true, preds, average="weighted")
    f1 = f1_score(y_true, preds, average="weighted")

    return {"precision": precision, "recall": recall, "f1_score": f1}


def compare_models(model1, model2, val_data):
    """Compare two models on the validation dataset."""
    metrics_model1 = evaluate_model(model1, val_data)
    metrics_model2 = evaluate_model(model2, val_data)

    comparison = {"model1": metrics_model1, "model2": metrics_model2}

    return comparison


def plot_comparison(comparison_results):
    """Plot a comparison of the metrics between two models."""
    # Code to generate plots (e.g., bar charts) comparing model metrics
    pass
