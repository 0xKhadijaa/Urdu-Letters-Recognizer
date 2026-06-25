# Urdu Handwritten Character Recognition System

## Overview

The Urdu Handwritten Character Recognition System is a Deep Learning-based Optical Character Recognition (OCR) project designed to recognize handwritten Urdu alphabets from images.

Unlike many OCR projects that rely on publicly available datasets, this project was built using a custom-created dataset consisting of thousands of handwritten Urdu character samples. The system leverages Deep Learning techniques to learn complex character patterns and accurately classify Urdu letters.

The project also includes a user-friendly Streamlit web application for real-time character prediction.

---

## Objectives

* Develop an intelligent system capable of recognizing handwritten Urdu characters.
* Build a custom dataset for Urdu alphabet classification.
* Compare the performance of Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN).
* Improve real-world robustness through extensive data augmentation.
* Deploy the trained model using Streamlit for interactive predictions.

---

## Problem Statement

Urdu handwritten text recognition is significantly more challenging than English character recognition due to:

* Complex character structures
* Similar-looking alphabets
* Variations in handwriting styles
* Differences in writing thickness and orientation

The goal of this project is to build a robust recognition system capable of accurately identifying handwritten Urdu alphabets despite these challenges.

---

## Dataset

### Custom Dataset

A custom dataset was created specifically for this project.

#### Dataset Statistics

* Total Classes: 38 Urdu Alphabets
* Images per Class: 500
* Total Images: 19,000
* Image Type: Handwritten Urdu Characters

Unlike many OCR systems that rely on standard datasets, this project focuses on real handwritten samples collected and organized for deep learning training.

---

## Data Augmentation

To improve model generalization and real-world performance, extensive augmentation techniques were applied.

### Augmentation Techniques

* Rotation
* Scaling
* Translation
* Brightness Variation
* Noise Injection
* Colored Background Augmentation

### Important Design Decision

Image flipping was intentionally avoided because flipping Urdu characters can alter their meaning and generate invalid training samples.

This helped preserve character integrity while still improving dataset diversity.

---

## Model Architectures

### Deep Neural Network (DNN)

A fully connected neural network was implemented as a baseline model.

### Convolutional Neural Network (CNN)

A custom CNN architecture was developed to better capture spatial features and visual patterns present in handwritten Urdu characters.

The CNN model included:

* Convolutional Layers
* Pooling Layers
* Batch Normalization
* Dropout Regularization
* Learning Rate Optimization
* Early Stopping

---

## Model Performance

### Accuracy Comparison

| Model | Accuracy |
| ----- | -------- |
| DNN   | 4%       |
| CNN   | 97%      |

### Key Observation

The CNN significantly outperformed the DNN because convolutional layers are better suited for image-based feature extraction and pattern recognition.

This experiment highlights the importance of using specialized architectures for computer vision tasks.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report
* Loss Curves
* Accuracy Curves

Performance visualization helped analyze class-wise prediction behavior and identify model strengths.

---

## Streamlit Application

A Streamlit-based web application was developed to allow users to interact with the trained model.

### Features

* Upload handwritten Urdu character images
* Real-time predictions
* User-friendly interface
* Fast inference using trained CNN model

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Machine Learning Utilities

* Scikit-Learn

## Installation

Clone the repository:

```bash
git clone https://github.com/0xKhadijaa/Urdu-Handwritten-Character-Recognition.git
```

Navigate to the project directory:

```bash
cd Urdu-Handwritten-Character-Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```
