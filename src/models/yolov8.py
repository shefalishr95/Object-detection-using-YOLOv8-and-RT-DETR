from ultralytics import YOLO
import mlflow
import os


class YOLOV8:
    """
    A class for managing the YOLOv8 model lifecycle, including training, validation, and exporting.

    This class simplifies the process of working with the YOLOv8 model by integrating training, validation,
    and model export functionalities, while also logging relevant metrics and artifacts with MLflow.

    Attributes:
    - model: An instance of the YOLOv8 model initialized with a specified model path.

    Methods:
    - train(kwargs, run_name): Trains the model using the provided keyword arguments and logs parameters and metrics.
    - validate(best_model_path, kwargs): Validates the model using the best model weights and logs validation metrics.
    - export_model(export_path, export_format): Exports the model in the specified format and logs the artifact.
    - run(train_kwargs, best_model_path, val_kwargs, export_format, export_path, run_name): Orchestrates the entire
      model lifecycle including training, validation, and exporting.
    """

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def train(self, kwargs, run_name="yolov8_experiment"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        with mlflow.start_run(run_name="yolov8_experiment") as run:
            mlflow.log_params(kwargs)
            results = self.model.train(**kwargs)
            mlflow.log_metrics(
                {
                    "Precision": results.results_dict["metrics/precision(B)"],
                    "Recall": results.results_dict["metrics/recall(B)"],
                    "mAP": results.results_dict["metrics/mAP50-95(B)"],
                    "mAP50": results.results_dict["metrics/mAP50(B)"],
                }
            )

    def validate(self, best_model_path, kwargs):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(**kwargs)
        mlflow.log_metrics(
            {
                "Validation_mAP": metrics.box.map,
                "Validation_mAP50": metrics.box.map50,
                "Validation_mAP75": metrics.box.map75,
            }
        )
        self.model = best_model

    def export_model(self, export_path, export_format="onnx"):
        self.model.export(format=export_format)
        mlflow.log_artifact(local_path=export_path)

    def run(
        self,
        train_kwargs,
        best_model_path,
        val_kwargs,
        export_format="onnx",
        export_path="best.onnx",
        run_name="yolov8_experiment",
    ):
        self.train(train_kwargs, run_name=run_name)
        self.validate(best_model_path, val_kwargs)
        self.export_model(export_path=export_path, export_format=export_format)
        mlflow.end_run()
