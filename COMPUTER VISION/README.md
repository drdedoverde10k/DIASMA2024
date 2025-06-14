# Chilean Coin Detection and Classification using CV-based Strategies

Coin identification and recognition systems can significantly improve the automation and efficiency of systems such as vending machines, public telephones, and coin counting machines. However, coin recognition presents a challenge in the fields of computer vision and machine learning due to varying rotations, scales, lighting conditions, and distinct surface patterns.

This project focuses on designing an efficient computer vision algorithm that is robust and invariant to rotation, translation, and scale—tailored specifically for the recognition of Chilean peso coins.

## Project Objectives
The main objective of this project is to develop a system capable of detecting and classifying Chilean coins in images without using convolutional neural networks (CNNs). The proposed solution must be able to:
- Segment image regions containing coins.
- Identify and determine the denomination of each visible coin (as long as its features are not severely occluded).

### Project Organization

```
.
├── dataset                                     : Contains all coins images
├── video                                       : Contains a video example
├── CMakeLists.txt                              : CMake instructions list
├── cam_calibration.py                          : Calibration for webcam script
├── opencv-coin-detection-and-counting.ipynb    : Test notebook
└── README.md                                   : Project Report
```
