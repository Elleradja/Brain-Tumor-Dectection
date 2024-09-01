# Brain-Tumor-Dectection
This code implements a Convolutional Neural Network (CNN) for binary classification, specifically designed to detect brain tumours from medical images. The overall structure of the code is as follows:

1. **Data Preparation**: 
   - The dataset is loaded and augmented using TensorFlow's `ImageDataGenerator` class. Data augmentation techniques such as rotation, zoom, and flips are applied to create variations of the images, which helps prevent overfitting and improves the model's generalization ability.
   - The data is split into training and validation sets using the `flow_from_directory` method, which automatically labels the images based on the directory structure.

2. **Model Architecture**:
   - A sequential CNN model is built with multiple convolutional layers followed by max-pooling layers. These layers are responsible for feature extraction from the images.
   - Dense layers are added after flattening the output from the convolutional layers. Dropout and L2 regularization are applied to reduce overfitting.
   - The model ends with a single neuron in the output layer with a sigmoid activation function, which is appropriate for binary classification.

3. **Model Compilation**:
   - The model is compiled using the Adam optimizer with a reduced learning rate to ensure a smooth and steady convergence.
   - The loss function used is binary cross-entropy, standard for binary classification tasks.
   - The model also tracks accuracy as a metric.

4. **Training the Model**:
   - The model is trained using the training data and validated using the validation data. Early stopping and learning rate reduction callbacks are implemented to prevent overfitting and to adjust the learning rate dynamically during training.
   - The training process is visualized by plotting accuracy and loss for both the training and validation sets, allowing for an assessment of the model's performance over time.

5. **Model Summary and Visualization**:
   - The model summary provides a detailed layer-by-layer overview, including the output shapes and the number of parameters.
   - Additionally, the architecture is visualized and saved as an image file for further review and documentation purposes.

This process ensures a robust training pipeline for brain tumour detection, focusing on both model performance and preventing overfitting.
---

### Model Summary

The model consists of a Convolutional Neural Network (CNN) architecture with the following layers:

- **Input Layer**: The input shape is defined as `(150, 150, 3)`, corresponding to the 150x150 pixel RGB images.
- **Convolutional Layers**: There are four convolutional layers with increasing filter sizes (32, 64, 128). Each convolutional layer is followed by a MaxPooling layer to reduce spatial dimensions while retaining the most important features.
- **Flatten Layer**: After the convolutional layers, the output is flattened into a one-dimensional vector.
- **Fully Connected Layers**: Two dense layers are used for classification, with 512 neurons in the first layer, followed by a Dropout layer to reduce overfitting. The output layer has a single neuron with a sigmoid activation function, suitable for binary classification.
- **Regularization**: Dropout and L2 regularization are used to prevent overfitting and ensure that the model generalizes well to unseen data.

The model summary details the output shape and the number of parameters for each layer, giving a comprehensive overview of the model's structure

### Training Accuracy Summary
Initially, the model started with lower accuracy, but as training continued, the model learned to distinguish between tumor and non-tumour images more effectively. The training accuracy showed a consistent upward trend, indicating that the model was successfully learning from the training data.
However, there were some fluctuations in accuracy, which are typical in deep learning, especially when training data is shuffled and when the model is fine-tuning its parameters. By the final epochs, the training accuracy stabilized, reaching a high value, suggesting that the model had learned the patterns in the training dataset well.
This high training accuracy is a positive sign, but it is crucial to compare it with the validation accuracy to ensure that the model is not overfitting and generalizes well to unseen data.
