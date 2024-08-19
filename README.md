# Traffic and Road Sign Detection using YOLOv8 and RT-DETR

## Overview

This project explores the application of two advanced object detection models, YOLOv8 and RT-DETR, to the task of traffic and road sign detection. Object detection is a key area in computer vision, essential for intelligent transportation systems and road safety. The focus is on comparing the performance of these models on a real-world dataset of traffic and road signs, with low-resolution images commonly used in training self-autonomous driving vehicles.

## Background and Problem Statement

Accurate and real-time detection of traffic and road signs is crucial for enhancing road safety and enabling autonomous driving technologies. YOLOv8, known for its speed and efficiency, is a benchmark in object detection tasks, while RT-DETR is a novel transformer-based model that claims superior performance. This project aims to compare these models using a publicly available dataset, evaluating their strengths and limitations in detecting traffic and road signs.

## Dataset

- **Name**: Traffic and Road Signs (Roboflow)
- **Size**: 10,000 images across 29 classes
- **Resolution**: 416x416
- **Split**: 7,092 training images, 1,884 validation images, 1,024 test images
- **Imbalance**: 9 underrepresented classes and 1 overrepresented class

The dataset was sourced from Roboflow and used without additional pre-processing.

## Model Overview

### YOLOv8

YOLOv8 is the latest iteration in the YOLO series, a CNN-based real-time object detection algorithm. It operates at 155 FPS with a mean average precision (mAP) of 52.7% in its non-enhanced version. YOLOv8 treats object detection as a single regression problem, predicting bounding boxes and probabilities for each region simultaneously.

### RT-DETR

RT-DETR, proposed by Lv et al. (2023), is a transformer-based model optimized for real-time object detection. It introduces a hybrid encoder and IoU-aware object query selection to enhance efficiency and accuracy. The model was tested using a pre-trained version on the COCO 2017 dataset due to limitations in custom dataset training.

## Model Performance

The models were trained and tested in the following environment:

- **Python Version**: 3.10.12
- **Torch Version**: 2.1.0 with CUDA support
- **Hardware**: Tesla T4 GPU, 12.7 GB RAM, 27.1/166.8 GB disk space

### Hyperparameters

| Hyperparameter          | YOLOv8 | RT-DETR |
|-------------------------|--------|---------|
| Epochs                  | 50     | 20      |
| Batch Size              | 16     | 16      |
| Image Size              | 416    | 416     |
| Optimizer               | Auto   | Auto    |
| Learning Rate           | Auto   | Auto    |
| Dropout Regularization  | 0.15   | 0.15    |

### Results

- **YOLOv8**: mAP50 of 0.288 and mAP50-95 of 0.239 after 50 epochs (1.432 hours).
- **RT-DETR**: mAP50 of 0.291 and mAP50-95 of 0.245 after 20 epochs (2.52 hours).

Both models struggled with underrepresented classes in the test set, and RT-DETR showed fluctuations in precision, likely due to insufficient training epochs.

## Conclusion

The initial results indicate that YOLOv8 offers faster training times, while RT-DETR may require more extensive training to fully leverage its capabilities. Future work will focus on refining the training process, particularly through dataset augmentation, to improve model accuracy and reliability.

## Sources

- Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., ... & Chen, J. (2024). Detrs beat yolos on real-time object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16965-16974).
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8.0.0) [Software]. Available at: https://github.com/ultralytics/ultralytics. License: AGPL-3.0. ORCID: 0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069.


## Appendix

1. **Architecture of YOLOv8**: [Link to YOLOv8 Architecture](https://yolov8.org/yolov8-architecture/)
2. **Architecture of RT-DETR**: Refer to Zhao et al. (2024)
