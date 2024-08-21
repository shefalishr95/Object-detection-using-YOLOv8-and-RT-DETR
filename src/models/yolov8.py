import mlflow
import mlflow.pytorch
from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self, model_path="yolov8n.pt"):
        # Initialize the model with a pre-trained weight file
        self.model = YOLO(model_path)

    def train(self, data, epochs, imgsz, device, batch_size=16, save=True, save_period=1, 
              project="YOLOv8_project", name="experiment", exist_ok=False, optimizer="AdamW", 
              seed=43, resume=True, fraction=1.0, lr0=1e-4, lrf=0.01):
        # Start MLflow run for tracking the training process
        with mlflow.start_run() as run:
            # Log parameters to MLflow
            mlflow.log_params({
                "data": data,
                "epochs": epochs,
                "imgsz": imgsz,
                "device": device,
                "batch_size": batch_size,
                "save": save,
                "save_period": save_period,
                "project": project,
                "name": name,
                "exist_ok": exist_ok,
                "optimizer": optimizer,
                "seed": seed,
                "resume": resume,
                "fraction": fraction,
                "lr0": lr0,
                "lrf": lrf
            })
            
            # Train the model
            results = self.model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                batch=batch_size,
                save=save,
                save_period=save_period,
                project=project,
                name=name,
                exist_ok=exist_ok,
                optimizer=optimizer,
                seed=seed,
                resume=resume,
                fraction=fraction,
                lr0=lr0,
                lrf=lrf
            )
            
            # Log key training metrics
            mlflow.log_metrics({
                "best_loss": results.best.loss,
                "best_map": results.best.map
            })
            
            # Save model checkpoints
            mlflow.pytorch.log_model(self.model, "model")

            # Custom logs (optional: loss curves, accuracy graphs, etc.)
            # Save artifacts like plots or reports
            # mlflow.log_artifact("path_to_artifacts")

            return results

    def evaluate(self, data):
        with mlflow.start_run() as run:
            # Evaluate the model and capture the results
            eval_results = self.model.val(data=data)

            # Log evaluation metrics
            mlflow.log_metrics({
                "mAP50": eval_results.metrics.map50,
                "mAP50-95": eval_results.metrics.map,
                "FPS": eval_results.speed.inference,
                "precision": eval_results.metrics.precision,
                "recall": eval_results.metrics.recall,
                "F1-Score": eval_results.metrics.f1
            })
            
            # Log confusion matrix and class-wise performance
            mlflow.log_artifact("confusion_matrix.png")
            mlflow.log_artifact("class_wise_performance.png")

            return eval_results

    def predict(self, image):
        with mlflow.start_run() as run:
            # Run inference and return the results
            results = self.model(image)
            
            # Log inference metrics, such as FPS
            mlflow.log_metric("inference_time", results.speed.inference)

            return results

# Example usage
if __name__ == "__main__":
    # Initialize the model
    yolo_model = YOLOv8Model()

    # Train the model with additional parameters
    train_results = yolo_model.train(
        data="coco8.yaml", 
        epochs=100, 
        imgsz=640, 
        device="cpu", 
        batch_size=16, 
        save=True, 
        save_period=1, 
        project="YOLOv8_project", 
        name="experiment_01", 
        exist_ok=True, 
        optimizer="AdamW", 
        seed=43, 
        resume=True, 
        fraction=0.8, 
        lr0=1e-4, 
        lrf=0.01
    )

    # Evaluate the model
    eval_results = yolo_model.evaluate(data="coco8.yaml")

    # Run inference
    predictions = yolo_model.predict("path/to/bus.jpg")
