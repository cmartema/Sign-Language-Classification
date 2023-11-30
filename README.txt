Uni: CJM2301

1. What is one-hot encoding? Why is this important and how do you implement it in keras?
    -It is a way to feed data to Machine learning models by converting categorical data into binary matrix.
     It is important because Machine Learning models expect input data to be numerical format. In Keras, you can 
     use the to_categorical function from the keras.utils module to perform one-hot encoding
2. What is dropout and how does it help overfitting?
    -Dropout is a regulazation technique for neural networks training used to prevent overfitting. It helps prevent overfitting by 
     training multiple models with different subsets of neurons active at each iteration. During inference, the predictions of all these models are averaged, 
     creating an ensemble effect.This ensemble averaging helps to smooth out predictions and reduce the impact of noisy or irrelevant features.

3. How does ReLU differ from the sigmoid activation function?
    -ReLU is a non-saturating function, which means that it does not become flat at the extremes of the input range. Instead, ReLU simply outputs 
     the input value if it is positive, or 0 if it is negative. The definition of a ReLU is h=max(0,a), where a=Wx+b and for sigmoid o(z) = 1/(1+e^(-z)).
     One major benefit is the reduced likelihood of the gradient to vanish. This arises when a>0. In this 
     regime the gradient has a constant value. In contrast, the gradient of sigmoids becomes increasingly small as the 
     absolute value of x increases. The constant gradient of ReLUs results in faster learning.

4. Why is the softmax function necessary in the output layer?
    -The softmax function is necessary in the output layer because it converts the raw output scores into a probability distribution over multiple classes.
     In the classification, the class with the highest probability is considered as the predicted class. This helps in making a definitive decision by highlighting 
     the class with the maximum probability.

5. This is a more practical calculation. Consider the following convolution network:
(a) Input image dimensions = 100x100x1
(b) Convolution layer with filters=16 and kernel size=(5,5)
(c) MaxPooling layer with pool size = (2,2) 
    What are the dimensions of the outputs of the convolution and max pooling layers?

    using the guide info from: https://cs231n.github.io/convolutional-networks/#conv

    w2 = (w1 - f + 2p)/s + 1
    h2 = (h1 - f +2p)/s+1
    d2 = k

    f = 5 --> Kernel size
    k = 16 --> # filters
    p = 0 --> zero padding
    s = 1 --> stride
    w1 = 100
    h1 = 100

    w2 = 96
    h2 = 96
    d2 = 16 

    96x96x16 -- > convolution layer

    Max pooling layer dimensions:
    w2 = w1/2
    h2 = h1/2

    48x48x16 --> max pooling dimensions

Brief Model:
The model optimizes feature extraction using convolutional and dense layers with ReLU activation. 
Spatial reduction is achieved through MaxPooling, complemented by dropout to prevent overfitting. 
Softmax activation suits multi-class classification. Highlights include data normalization, a validation split, 
and limited training epochs for efficiency and reproducibility. The architecture strikes a refinement balance between 
complexity and interpretability, catering to diverse dataset requirements.