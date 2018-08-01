# Shallow Neural Networks 

+ The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. [check this post](https://stats.stackexchange.com/questions/101560/tanh-activation-function-vs-sigmoid-activation-function#101563)

> There are two reasons for that choice (assuming you have normalized your data, and this is very important):
Having stronger gradients: since data is centered around 0, the derivatives are higher. To see this, calculate the derivative of the tanh function and notice that its range (output values) is [0,1].
The range of the tanh function is [-1,1] and that of the sigmoid function is [0,1].
Avoiding bias in the gradients. This is explained very well in the paper, and it is worth reading it to understand these issues.

+ You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer? (Answer: Sigmoid)
> Note: The output value from a sigmoid function can be easily understood as a probability.
Sigmoid outputs a value between 0 and 1 which makes it a very good choice for binary classification. You can classify as 0 if the output is less than 0.5 and classify as 1 if the output is more than 0.5. It can be done with tanh as well but it is less convenient as the output is between -1 and 1.

