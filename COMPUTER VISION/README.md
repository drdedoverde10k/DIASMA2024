# Chilean Coin Detection and Classification using CV-based Strategies
Real-time coin detector and counter using classic Computer Vision techniques

Coin identification and recognition systems can significantly improve the automation and efficiency of systems such as vending machines, public telephones, and coin counting machines. However, coin recognition presents a challenge in the fields of computer vision and machine learning due to varying rotations, scales, lighting conditions, and distinct surface patterns.

This project focuses on designing an efficient computer vision algorithm that is robust and invariant to rotation, translation, and scale—tailored specifically for the recognition of Chilean peso coins.

## Project Objectives
This project was developed for the **Computer Vision** course in the Ph.D. program *Applied Informatics for Health and the Environment* (UTEM). Its goal is to **detect, classify, and count** Chilean coins in real time using only classic OpenCV tools—no deep-learning models.

The proposed solution must be able to:
- Segment image regions containing coins.
- Identify and determine the denomination of each visible coin (as long as its features are not severely occluded).

<p align="center">
  <img src="video/vid1.gif" alt="Chilean Coin Detection" width="600">
</p>

<p align="center">
  <em>
    Matching of Chilean peso coins.
  </em>
</p>

## Features
- 📸 **Real-time processing** from a webcam (720 p).  
- 🔵 **Automatic metric calibration** via a standard credit card (85.6 mm × 53.98 mm).  
- 🟢 **Hough Circle Transform** for initial detection.  
- 🟡 **Heuristic filtering** by circularity, diameter (mm) and HSV hue for each denomination.  
- 🔴 **False-positive rejection** (buttons, tokens) by searching for inner circles.  
- 📝 On-screen overlay with total and per-denomination counts ($1 / $5, $10, $50, $100, old $100, $500).  
- 🗂️ Test set covering various backgrounds (`bkx`, `wx`, `blurx`), a fake coin (`pkx`), and many coins randomly situated (`mcx`).

## Coin Detection Results

ratio =  predicted/real

### Table 1 – Overall Counting Summary

| Denomination      | Predicted | Ground-truth | Error | Ratio |
|-------------------|----------:|-------------:|------:|------:|
| \$1 / \$5         | 34 | 35 | −1  | 0.97 |
| \$10              | 96 | 183 | −87 | 0.52 |
| \$50              | 60 | 139 | −79 | 0.43 |
| \$100             | 82 | 174 | −92 | 0.47 |
| \$100 (Old)       | 33 | 0  | +33 | — |
| \$500             | 106 | 154 | −48 | 0.69 |
| ?? (Not Coin)     | 237 | 23 | +214 | 10.30 |

### Table 2 – Per-Denomination Performance Statistics (51 images)

| Denomination      | Mean Ratio | Std. Dev. | Median | Accuracy |
|-------------------|-----------:|----------:|-------:|---------:|
| \$1 / \$5         | 0.91 | 0.77 | 1.00 | 51.1 % |
| \$10              | 0.68 | 1.38 | 0.50 | 13.7 % |
| \$50              | 0.43 | 0.38 | 0.33 | 7.8 % |
| \$100             | 0.43 | 0.32 | 0.50 | 7.8 % |
| \$100 (Old)       | 1.00 | 0.00 | 1.00 | 100.0 % |
| \$500             | 1.21 | 1.30 | 1.00 | 19.6 % |



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

## Requirements
| Package | Min version |
|---------|-------------|
| Python  | 3.9 |
| OpenCV  | 4.5 |
| NumPy   | 1.19 |
| Matplotlib (optional, debugging) | 3.4 |
> **Tip:** Use a virtual environment (venv or conda).

## Installation
```bash
git clone https://github.com/drdedoverde10k/DIASMA2024.git
cd "DIASMA2024/COMPUTER VISION"
python -m venv .venv
source .venv/bin/activate
```
## Launch App
```bash
python chilean_coin_detection.py

# Press 'q' to close/exit

```
