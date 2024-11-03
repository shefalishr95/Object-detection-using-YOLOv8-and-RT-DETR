# Evaluating State-of-the-Art Object Detection Models for Auto-Vision 

## Project Overview

This project explores the application of two advanced object detection models, YOLOv8 and RT-DETR, to traffic and road sign detection. Object detection is crucial for intelligent transportation systems and road safety. This project compares these models using MLflow on a real-world dataset of traffic and road signs, with low-resolution images often used in autonomous vehicle training.

## Table of Contents

1. [Background and Dataset](#background-and-dataset)
2. [Models](#models)
3. [Experiments](#experiments)
4. [Results](#results)
5. [Inference](#inference)
6. [Limitations](#limitations)
7. [License](#license)
8. [Sources](#sources)
9. [Appendix](#appendix)

## Background and Dataset

Accurate and real-time detection of traffic and road signs is vital for enhancing road safety and enabling autonomous driving technologies. YOLOv8, known for its speed and efficiency, is a benchmark in object detection, while RT-DETR is a novel transformer-based model with claimed superior performance. This project aims to compare these models using a publicly available dataset, evaluating their strengths and limitations in detecting traffic and road signs.

## Dataset

- **Name**: Traffic and Road Signs (Roboflow)
- **Size**: 10,000 images across 29 classes
- **Resolution**: 416x416
- **Split**: 7,092 training images, 1,884 validation images, 1,024 test images
- **Imbalance**: 9 underrepresented classes and 1 overrepresented class (not addressed in this project)

## Models

1. **YOLOv8**

YOLOv8 is the latest iteration in the YOLO series, a CNN-based real-time object detection algorithm. It treats object detection as a single regression problem, predicting bounding boxes and probabilities for each region simultaneously.

![image](https://github.com/user-attachments/assets/b1fe031a-0f96-49af-b9b1-8f0ca3a3d3ea)

2. **RT-DETR**

RT-DETR, proposed by Lv et al. (2023), is a transformer-based model optimized for real-time object detection. It introduces a hybrid encoder and IoU-aware object query selection to enhance efficiency and accuracy.

![image](https://github.com/user-attachments/assets/b8bbb01b-db20-4eb8-a0df-123ed1e16d4d)

## Experiments

A series of experiments were conducted to evaluate YOLOv8 and RT-DETR models for traffic and road sign detection. The experiments focused on comparing performance across different configurations.

### Experiment Design

The experiments included the following configurations:

1. **48 epochs, Batch size 16**: Initial configuration to evaluate model performance.
2. **64 epochs, Batch size 16**: Extended training with the same batch size.
3. **64 epochs, Batch size 32**: Training with a larger batch size.

The models were trained and tested in the following environment:

- **Python Version**: 3.10.12
- **Torch Version**: 2.1.0 with CUDA support
- **Hardware**: Tesla T4 GPU, 12.7 GB RAM, 27.1/166.8 GB disk space

### Tracking and Benchmarking

MLflow was used for tracking and benchmarking the experiments:

- **MLflow Tracking**: Parameters, metrics (model and system), and artifacts were logged for each experiment.
- **Metrics Tracked**: Metrics logged included:

  - **mAP (Mean Average Precision)**: Overall detection accuracy.
  - **mAP50 and mAP50-95**: mAP values at different IoU thresholds.
  - **Confusion Matrix**: Prediction errors.
  - **Precision and Recall**: Measures of the model's identification and classification performance.

- **Benchmarking**: Results from each configuration were compared to determine the best performing setup. YOLOv8 and RT-DETR models were assessed based on mAP scores, precision and recall.

Due to resource limitations, comprehensive tuning of hyperparameters such as optimizers and learning rates was not feasible.

Results can be found at: https://dagshub.com/shefali.0695/Object-detection-using-YOLOv8-and-RT-DETR.mlflow

## Results

### YOLOv8 Results

| Configuration    |   mAP   |  mAP50  |  mAP75  | Mean Precision | Mean Recall |
| ---------------- | ------- | ------- | ------- | -------------- | ----------- |
| Baseline         | 0.0996  | 0.1812  | 0.1235  | -              | -           |
| 48 epochs, BS 16 | 0.2302  | 0.2800  | 0.2767  | 0.2705         | 0.2523      |
| 64 epochs, BS 16 | 0.2388  | 0.2842  | 0.2818  | 0.2223         | 0.2904      |
| 64 epochs, BS 32 | 0.2365  | 0.2831  | 0.2806  | 0.2097         | 0.2827      |

### RT-DETR Results

| Configuration    | mAP    | mAP50  | mAP75  | Mean Precision | Mean Recall |
| ---------------- | ------ | ------ | ------ | -------------- | ----------- |
| Baseline         | 0.0996 | 0.1812 | 0.1235 | -              | -           |
| 48 epochs, BS 16 | 0.2302 | 0.2800 | 0.2767 | 0.1931         | 0.3019      |
| 64 epochs, BS 16 | 0.2388 | 0.2842 | 0.2818 | 0.1964         | 0.3000      |
| 64 epochs, BS 32 | 0.0780 | 0.0918 | 0.0900 | 0.0730         | 0.2738      |

## Inference

Only two hyperparameters were tested:
  - Number of epochs (48 vs 64)
  - Batch size (16 vs 32)
Other parameters remained at default values. Conclusions are specifically limited to these variations

1. **Observed Patterns**
- Epochs Effect:
  - Both models improved with longer training (48 → 64 epochs)
  - Similar improvement magnitude for both models. 
  
- Batch Size Effect:
  - YOLOv8: Relatively stable with BS change
    - Only minor drop from BS 16 (0.2388 mAP) to BS 32 (0.2365 mAP)
  - RT-DETR: Highly sensitive to BS change
    - Significant drop from BS 16 (0.2388 mAP) to BS 32 (0.0780 mAP)

2. **Key Findings**
- Both models reach identical best performance (mAP: 0.2388) with:
  - 64 epochs
  - Batch size 16
- Main difference is in batch size sensitivity within the constraints of this experiment

## Limitations

1. **Hyperparameter Exploration**
   - Limited testing to only two hyperparameters (epochs and batch size) due to resource constraints

2. **Dataset Characteristics**
   - Class imbalance: 9 underrepresented classes and 1 overrepresented class - apply class-specific augmentation?
   - Limited dataset size (10,000 images) may affect model generalization

3. **Computational Resources**
   - Experiments constrained by available GPU resources (Tesla T4)
   - Limited RAM (12.7 GB) restricted batch size options

4. **Evaluation Metrics**
   - Focus on inference time and computational efficiency measurements in the future and other object detection specific metrics like Intersection over Union (IoU), Frame Rate (FPS), object size sensitivity analysis, detection confidence scores distribution, miss rate across different occlusion levels, and multi-scale detection performance. 
   - Additionally, evaluate model robustness through testing on edge cases such as partially visible signs, varying lighting conditions, and motion blur scenarios.

5. **Training Duration**
   - Maximum training limited to 64 epochs
   - Potential for improved performance with extended training not explored (incomplete investigation of convergence patterns)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Sources

- Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., ... & Chen, J. (2024). DETRs Beat YOLOs on Real-Time Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16965-16974).
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Software]. Available at: https://github.com/ultralytics/ultralytics. License: AGPL-3.0.

## Appendix

1. **Architecture of YOLOv8**: [Link to YOLOv8 Architecture](https://yolov8.org/yolov8-architecture/)
2. **Architecture of RT-DETR**: Refer to Zhao et al. (2024)
