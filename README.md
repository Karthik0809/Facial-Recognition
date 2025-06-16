# Facial Recognition using CNN

This repository contains a deep learning-based facial recognition model using a Convolutional Neural Network (CNN). The model is trained on the **Labeled Faces in the Wild (LFW)** dataset to classify individuals based on their facial images.

## Dataset

The dataset used in this project is the **Labeled Faces in the Wild (LFW)** dataset. It consists of grayscale images of various celebrities, each labeled with the individual's name. 

- **Total images:** 1140  
- **Image size:** 62x47 pixels  
- **Number of individuals:** 12  
- **Minimum images per person:** 100  

## Model Architecture

The model is built using a **Convolutional Neural Network (CNN)** with the following architecture:

- **Input Layer:** 62×47 grayscale images
- **Convolutional Layer 1:** 32 filters, ReLU activation
- **Max Pooling Layer 1**
- **Convolutional Layer 2:** 64 filters, ReLU activation
- **Max Pooling Layer 2**
- **Dropout:** 25%
- **Fully Connected Layer:** 512 neurons with ReLU activation
- **Dropout:** 50%
- **Output Layer:** Softmax activation for classification

The model is compiled using:
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy

## Training Details

- **Train-Test Split:** 80%-20%
- **Batch Size:** 32
- **Epochs:** Up to 100 with early stopping (patience 5)
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

2. Run the Python script:
   ```sh
   python facial_recognition.py
   ```

## Web Deployment using FastAPI

An interactive web interface is provided using **FastAPI**. It allows you to
upload an image, performs prediction using the trained model and displays the
result.

To start the web server:

```sh
pip install fastapi uvicorn[standard] pillow scikit-learn tensorflow matplotlib seaborn
python fastapi_app.py
```

Then navigate to `http://localhost:8000` in your browser to try the demo.

## Future Improvements

- Implement data augmentation for better generalization
- Experiment with deeper CNN architectures
- Optimize hyperparameters for improving the accuracy
- Package the model so the FastAPI app can load pre-trained weights

## Author

- **Karthik Mulugu**  

