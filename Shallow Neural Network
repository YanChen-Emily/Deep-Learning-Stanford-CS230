# Shallow Neural Networks 

(Week 3 quiz from Coursera Deep Learning Course)

+ The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. [check this post](https://stats.stackexchange.com/questions/101560/tanh-activation-function-vs-sigmoid-activation-function#101563)

> There are two reasons for that choice (assuming you have normalized your data, and this is very important):
Having stronger gradients: since data is centered around 0, the derivatives are higher. To see this, calculate the derivative of the tanh function and notice that its range (output values) is [0,1].
The range of the tanh function is [-1,1] and that of the sigmoid function is [0,1].
Avoiding bias in the gradients. This is explained very well in the paper, and it is worth reading it to understand these issues.

+ You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer? (Answer: Sigmoid)
> Note: The output value from a sigmoid function can be easily understood as a probability.
Sigmoid outputs a value between 0 and 1 which makes it a very good choice for binary classification. You can classify as 0 if the output is less than 0.5 and classify as 1 if the output is more than 0.5. It can be done with tanh as well but it is less convenient as the output is between -1 and 1.

+  
```python
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
```
What will be B.shape? (B.shape = (4, 1))
> we use (keepdims = True) to make sure that A.shape is (4,1) and not (4, ). It makes our code more rigorous.

+ Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons. 
> However, Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector.

+ You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?

> This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow. Note that tanh becomes flat for large values, this leads its gradient to be close to zero. This slows down the optimization algorithm.
