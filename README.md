# Benchmarking State-of-the-Art Object Detection Models for Autonomous Vehicles

## Project Overview

This project explores the application of two advanced object detection models, YOLOv8 and RT-DETR, to traffic and road sign detection. Object detection is crucial for intelligent transportation systems and road safety. This project compares these models using MLflow on a real-world dataset of traffic and road signs, with low-resolution images often used in autonomous vehicle training.

## Table of Contents

1. [Background and Dataset](#background-and-dataset)
2. [Models](#models)
3. [Experiments](#experiments)
4. [Results](#results)
5. [Usage](#usage)
6. [License](#license)
7. [Sources](#sources)
8. [Appendix](#appendix)

## Background and Dataset

Accurate and real-time detection of traffic and road signs is vital for enhancing road safety and enabling autonomous driving technologies. YOLOv8, known for its speed and efficiency, is a benchmark in object detection, while RT-DETR is a novel transformer-based model with claimed superior performance. This project aims to compare these models using a publicly available dataset, evaluating their strengths and limitations in detecting traffic and road signs.

## Dataset

- **Name**: Traffic and Road Signs (Roboflow)
- **Size**: 10,000 images across 29 classes
- **Resolution**: 416x416
- **Split**: 7,092 training images, 1,884 validation images, 1,024 test images
- **Imbalance**: 9 underrepresented classes and 1 overrepresented class

## Models

1. **YOLOv8**

YOLOv8 is the latest iteration in the YOLO series, a CNN-based real-time object detection algorithm. It treats object detection as a single regression problem, predicting bounding boxes and probabilities for each region simultaneously.

2. **RT-DETR**

RT-DETR, proposed by Lv et al. (2023), is a transformer-based model optimized for real-time object detection. It introduces a hybrid encoder and IoU-aware object query selection to enhance efficiency and accuracy.

## Experiments

Certainly! Here is a revised version of the **Experiments** section without adjectives:

---

## Experiments

A series of experiments were conducted to evaluate YOLOv8 and RT-DETR models for traffic and road sign detection. The experiments focused on comparing performance across different configurations.

### Experiment Design

The experiments included the following configurations:

1. **48 epochs, Batch size 16**: Initial configuration to evaluate model performance.
2. **64 epochs, Batch size 16**: Extended training with the same batch size.
3. **64 epochs, Batch size 32**: Training with a larger batch size.
4. **100 epochs, Batch size 32**: Ongoing experiment with additional epochs.

These configurations were designed to assess the effects of training duration and batch size on model performance.

### Tracking and Benchmarking

MLflow was used for tracking and benchmarking the experiments:

- **MLflow Tracking**: Parameters, metrics, and artifacts were logged for each experiment. Parameters such as learning rate, batch size, and number of epochs were recorded.
- **Metrics Tracked**: Metrics logged included:

  - **mAP (Mean Average Precision)**: Overall detection accuracy.
  - **mAP50 and mAP50-95**: mAP values at different IoU thresholds.
  - **Inference Time (FPS)**: Real-time detection capability.
  - **Class-wise Performance**: Performance across individual classes.
  - **Confusion Matrix**: Prediction errors.
  - **Precision and Recall**: Measures of the model's identification and classification performance.

- **Benchmarking**: Results from each configuration were compared to determine the best performing setup. YOLOv8 and RT-DETR models were assessed based on mAP scores, precision, recall, and other metrics.

Future work will include adding inference metrics to evaluate real-world performance, such as speed and efficiency. Detailed results and visualizations will be available on the MLflow UI.

## Results

### YOLOv8 Results

| Configuration    | mAP    | mAP50  | mAP75  | Precision | Recall |
| ---------------- | ------ | ------ | ------ | --------- | ------ |
| Baseline         | 0.0996 | 0.1812 | 0.1235 | 0.11      | 0.09   |
| 48 epochs, BS 16 | 0.2302 | 0.2800 | 0.2767 | 0.24      | 0.21   |
| 64 epochs, BS 16 | 0.2388 | 0.2842 | 0.2818 | 0.26      | 0.23   |
| 64 epochs, BS 32 | 0.2365 | 0.2831 | 0.2806 | 0.25      | 0.22   |

### RT-DETR Results

| Configuration    | mAP    | mAP50  | mAP75  | Precision | Recall |
| ---------------- | ------ | ------ | ------ | --------- | ------ |
| Baseline         | 0.0100 | 0.0403 | 0.0004 | 0.02      | 0.01   |
| 48 epochs, BS 16 | 0.1939 | 0.2303 | 0.2303 | 0.20      | 0.18   |
| 64 epochs, BS 16 | 0.2012 | 0.2351 | 0.2335 | 0.22      | 0.19   |
| 64 epochs, BS 32 | 0.1998 | 0.2340 | 0.2324 | 0.21      | 0.18   |

The YOLOv8 model showed consistent improvement across different configurations, with mAP increasing from 0.0996 at baseline to 0.2388 with 64 epochs and batch size 16. Precision and Recall also improved across configurations. Inference metrics will be added in future evaluations to provide a more comprehensive view of the model's real-world performance, including speed and efficiency during deployment. Detailed results and visualizations will be accessible on the MLflow UI soon.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Sources

- Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., ... & Chen, J. (2024). DETRs Beat YOLOs on Real-Time Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16965-16974).
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Software]. Available at: https://github.com/ultralytics/ultralytics. License: AGPL-3.0.

## Appendix

1. **Architecture of YOLOv8**: [Link to YOLOv8 Architecture](https://yolov8.org/yolov8-architecture/)
2. **Architecture of RT-DETR**: Refer to Zhao et al. (2024)
