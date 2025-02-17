# Convolutional Neural Network Optimization for MNIST

## Project Overview
This project focuses on designing an optimized Convolutional Neural Network (CNN) to improve classification performance on the MNIST dataset. The goal was to analyze a predefined 3-layer model, identify its limitations, and propose enhancements that improve accuracy and efficiency.

## Dataset
- **MNIST Dataset**: A collection of grayscale images of hand-written digits (0-9).
- **Preprocessing**: Images were normalized and converted into a format suitable for training the neural network.

## Initial Model
The baseline model consisted of:
- A **3-layer CNN** with:
  - **Convolutional layer** (Feature extraction)
  - **Flattening layer** (Transforming multi-dimensional input into 1D)
  - **Dense layer** (Final classification)
- Various optimizers were tested (**SGD, Adam, Adadelta**) to evaluate their impact on model performance.

### Limitations of Initial Model
- Shallow architecture limiting feature extraction
- Absence of batch normalization and pooling layers
- Risk of overfitting due to lack of regularization
- Slow convergence and suboptimal accuracy

## Enhanced Model
To address the limitations, a deeper CNN inspired by VGG architecture was implemented:
- **6 Convolutional Layers** with increasing filter sizes (64, 128)
- **MaxPooling Layers** for down-sampling to reduce computational cost
- **Batch Normalization** for stabilizing training
- **Dropout Layers** to prevent overfitting
- **Fully Connected Layers** for classification
- **Data Augmentation** (Random Zoom) to enhance generalization
- **Regularization Techniques**: Early Stopping and Reduce Learning Rate on Plateau

## Training and Hyperparameter Tuning
- **Optimizer**: Adam (Selected due to superior convergence and stability)
- **Epochs**: 120 (with Early Stopping)
- **Batch Size**: 128
- **Validation Split**: 10%
- **Learning Rate Scheduling**: Reduce Learning Rate on Plateau to adapt learning pace

## Results
| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|--------|------------------|---------------------|--------------|
| Baseline (3-layer) | 93.57% | 94.25% | 94.04% |
| Enhanced (6-layer) | 97.76% | 95.25% | 94.66% |

- Improved accuracy with enhanced model
- Smoother convergence and better generalization
- Increased training time due to deeper architecture and additional layers

## Conclusion
This project demonstrated the effectiveness of deeper architectures, data augmentation, and training optimizations in improving CNN performance. The final model achieved a **significant improvement in stability and accuracy** while mitigating overfitting through regularization techniques.

## Running the Code
### Prerequisites
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib

### Execution
1. Load the MNIST dataset.
2. Preprocess the data (normalize and reshape).
3. Train the initial CNN model and evaluate performance.
4. Implement the enhanced CNN model.
5. Train with hyperparameter tuning and observe performance metrics.
6. Plot accuracy and loss curves.
7. Save and load the trained model for inference.

## References
- [Keras Documentation](https://keras.io)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Machine Learning Techniques](https://machinecurve.com)

---
Developed by **Gabriel Pecoraro**


