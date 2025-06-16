# Facial Recognition using CNN

This repository contains a deep learning-based facial recognition model using a Convolutional Neural Network (CNN). The model is trained on the **Labeled Faces in the Wild (LFW)** dataset to classify individuals based on their facial images.

## Dataset

The dataset used in this project is the **Labeled Faces in the Wild (LFW)** dataset. It consists of grayscale images of various celebrities, each labeled with the individual's name. 

- **Total images:** 1140  
- **Image size:** 62x47 pixels  
- **Number of individuals:** 12  
- **Minimum images per person:** 100  

## Model Architecture

The project uses a deeper **Convolutional Neural Network (CNN)** built with three convolutional blocks followed by fully connected layers:

- **Input Layer:** 62×47 grayscale images
- **Conv Block 1:** 32 filters + Max Pooling
- **Conv Block 2:** 64 filters + Max Pooling
- **Conv Block 3:** 128 filters + Max Pooling
- **Dropout:** 30%
- **Dense Layer:** 256 neurons with ReLU
- **Dropout:** 50%
- **Output Layer:** Softmax activation for classification

The model is compiled using:
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

## Training Details

- **Train-Test Split:** 80%-20%
- **Batch Size:** 32
- **Epochs:** Up to 50 with early stopping (patience 5)
- **Data Augmentation:** Rotation, shift, and horizontal flip

### Training and Validation Accuracy

| Metric       | Accuracy |
|-------------|----------|
| Training Accuracy | **~95%** |
| Validation Accuracy | **~85%** |

## Results & Evaluation

The model's performance was evaluated using a classification report and a confusion matrix. Key evaluation metrics include:

- **Precision & Recall**: High for most individuals
- **Confusion Matrix**: Shows misclassifications for visually similar faces

### Sample Prediction

A random test image is selected, and the model predicts the person's identity with high accuracy.

## How to Run the Code

1. Clone the repository:
   ```sh
   git clone https://github.com/Karthik0809/facial-recognition.git
   cd facial-recognition
   ```

2. Train the model (this will create `model.h5`):
   ```sh
   python facial_recognition.py
   ```

## Web Deployment using FastAPI

An interactive web interface is provided using **FastAPI**. It allows you to
upload an image, performs prediction using the trained model and displays the
result.

To start the web server after training:

```sh
pip install fastapi uvicorn[standard] pillow scikit-learn tensorflow matplotlib seaborn
python fastapi_app.py
```

Then navigate to `http://localhost:8000` in your browser to try the demo.

## Features Added

- Data augmentation for better generalization
- Deeper CNN architecture with three convolutional blocks
- Early stopping and model checkpointing for optimized training
- FastAPI app loads pre-trained `model.h5`
- Web interface performs optional face detection and cropping

## Author

- **Karthik Mulugu**  

