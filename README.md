# Table Detection Model

This repository contains a deep learning model for detecting tables in images and predicting their bounding boxes. The model is built using TensorFlow/Keras and follows a regression approach for predicting bounding box coordinates.

## Components

### 1. Data Generator (DataGenerator)
- Loads and preprocesses images and bounding box annotations (from JSON files) in batches for training.
- Normalizes images and bounding boxes for model input.

### 2. CNN Model (create_cnn_model)
- A simple CNN architecture with convolutional layers, max-pooling, and fully connected layers.
- Outputs 4 values representing the bounding box coordinates: `[x_min, y_min, x_max, y_max]`.

### 3. Model Training (train_table_detection_model)
- Trains the CNN model using images and bounding box annotations.
- Uses Mean Squared Error (MSE) loss and Adam optimizer.
- Saves the trained model as a `.h5` file.

### 4. Bounding Box Prediction (predict_bboxes)
- Loads test images, resizes, normalizes, and predicts bounding boxes.
- Saves predictions in a CSV file.

## Training Results
- **Training Loss:** 0.44  
- **Validation Loss:** 0.479

## Requirements
- TensorFlow
- NumPy
- Pillow
- pandas (for CSV output)

## Usage

### 1. Train the Model
Set paths for training and validation images/annotations, then call `train_table_detection_model`:


### 2. Predict Bounding Boxes
Use `predict_bboxes` with the trained model to predict bounding boxes for test images:


