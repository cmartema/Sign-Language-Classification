# Sign Language Classification Assignment

## Student Information

- **Name:** Your Name
- **UNI:** Your UNI

## Answers to Questions

1. **One-hot encoding:** One-hot encoding is...
2. **Dropout:** Dropout is...
3. **ReLU vs. Sigmoid:** ReLU is..., while Sigmoid is...
4. **Softmax in Output Layer:** Softmax is necessary...
5. **Convolution and MaxPooling Layer Outputs:**
   - Convolution layer output dimensions: ...
   - MaxPooling layer output dimensions: ...

## Brief Explanation of Architecture Choices

1. **Convolutional Layers:** Utilized to capture spatial hierarchies in images, extracting features through filters.
2. **MaxPooling Layers:** Downsample feature maps, reducing computational complexity and focusing on important features.
3. **Flatten Layer:** Flattens the output for input to fully connected layers.
4. **Dense Layers:** Fully connected layers for classification.
5. **ReLU Activation:** Efficiently handles non-linearity, aiding the network's ability to learn complex patterns.
6. **Softmax Activation in Output Layer:** Essential for multi-class classification, providing class probabilities.

## Workflow Information

1. **Data Preprocessing:** Normalize pixel values and reshape images.
2. **Model Training:** Utilize Keras' `fit` method, incorporating hyperparameters like batch size and epochs.
3. **Model Prediction:** Implement resizing/downsampling for test images and use the trained model for predictions.
4. **Visualization:** Provided methods to visualize data and accuracy trends.

## README Submission Instructions

1. **Testing:**
   - Before submitting, restart the runtime and run all cells to ensure reproducibility.

2. **Grading:**
   - Models achieving 90% accuracy in the test set will receive a full score. The grading algorithm considers accuracy with a maximum of 90% for calculation.

---

**Note:** Replace placeholders such as `Your Name` and `Your UNI` with your actual name and UNI. Fill in the answers to the questions and provide a brief explanation of your architecture choices and workflow. Adjust hyperparameters as needed.
