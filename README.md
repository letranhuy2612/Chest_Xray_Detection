# Chest_Xray_Detection

## Overview
Dataset:For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "14 1 0 0 1 1" (14 is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

## Working
![Screenshot 2022-05-05 125905](https://user-images.githubusercontent.com/85574548/166877366-e502c49d-fa8b-4493-b6af-44bcb2bea8ee.png)
Results are definitely wrong due to new train 5 epochs
Convert file weights pytorch to onnx 
Load file weights onnx with OpenCV and deploy web using Streamlit
