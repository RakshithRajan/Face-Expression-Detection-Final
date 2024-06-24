# Face Emotion Recognition Project

This project explores two models for recognizing facial expressions from images: ResNet50 and a custom Convolutional Neural Network (CNN). The goal is to accurately classify facial emotions such as anger, disgust, fear, happiness, neutral, sadness, and surprise.

## ResNet50 Model

ResNet50 (Residual Network 50) is a deep convolutional neural network architecture introduced by Kaiming He et al. in 2015. It is renowned for its depth and ability to effectively train very deep networks.

### Features of ResNet50:
- **Depth**: ResNet50 consists of 50 layers, hence the name, making it deeper than traditional CNN architectures. These layers are organized into blocks, with residual connections allowing gradients to flow more directly.
- **Pre-trained on ImageNet**: Typically, ResNet50 is pre-trained on the ImageNet dataset, leveraging a large number of diverse images across various categories to learn general features.
- **Global Average Pooling**: Utilizes global average pooling before the final classification layer, reducing the number of parameters and aiding in generalization.
- **Softmax Activation**: Outputs probabilities for each emotion class, making it suitable for multi-class classification tasks.

## Custom CNN Model

The custom CNN model was developed to address the limitations encountered with ResNet50 for this specific facial emotion recognition task. Here’s a detailed breakdown of the custom CNN model architecture based on the provided code:

### Custom CNN Model Architecture:
- **Input Layer**: 
  - Accepts grayscale images of size 48x48 pixels (`input_shape=(48, 48, 1)`).

- **Convolutional Layers**:
  - **1st Convolutional Layer**: 
    - Applies 64 filters of size 3x3 with ReLU activation.
    - Followed by MaxPooling (2x2), Batch Normalization, and Dropout (25%).
  
  - **2nd Convolutional Layer**: 
    - Applies 128 filters of size 5x5 with ReLU activation.
    - Followed by MaxPooling (2x2), Batch Normalization, and Dropout (25%).

  - **3rd Convolutional Layer**: 
    - Applies 512 filters of size 3x3 with ReLU activation.
    - Followed by MaxPooling (2x2), Batch Normalization, and Dropout (25%).

  - **4th Convolutional Layer**: 
    - Applies 512 filters of size 3x3 with ReLU activation.
    - Followed by MaxPooling (2x2), Batch Normalization, and Dropout (25%).

- **Flatten Layer**: 
  - Converts the 2D feature maps into a 1D vector for input to the fully connected layers.

- **Fully Connected Layers**:
  - **1st Dense Layer**: 
    - Consists of 256 neurons with ReLU activation, followed by Batch Normalization and Dropout (25%).
  
  - **2nd Dense Layer**: 
    - Consists of 512 neurons with ReLU activation, followed by Batch Normalization and Dropout (25%).
  
- **Output Layer**: 
  - Uses a Dense layer with 7 neurons (one for each emotion class) and softmax activation to output probabilities.

### Model Training and Performance:
- Initially, ResNet50 achieved an accuracy of approximately 40% on the validation dataset.
- The custom CNN model, designed specifically for this task, achieved a significant improvement, reaching a validation accuracy of 61%.
- This performance gain underscores the effectiveness of tailoring the model architecture to better capture and classify facial emotions from the given dataset.

## Dataset

The dataset comprises grayscale 48x48 pixel images of faces labeled with one of seven emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The dataset is divided into training and validation sets for model training and evaluation.

## Requirements

Ensure you have the following Python libraries installed to run the project:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Usage

1. Clone the repository.
2. Install the required libraries.
3. Use the provided scripts to train and evaluate both the ResNet50 and custom CNN models.
