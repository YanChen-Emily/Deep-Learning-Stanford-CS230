# Shallow Neural Networks 

+ The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. [check this post](https://stats.stackexchange.com/questions/101560/tanh-activation-function-vs-sigmoid-activation-function#101563)

There are two reasons for that choice (assuming you have normalized your data, and this is very important):
Having stronger gradients: since data is centered around 0, the derivatives are higher. To see this, calculate the derivative of the tanh function and notice that its range (output values) is [0,1].
The range of the tanh function is [-1,1] and that of the sigmoid function is [0,1].
Avoiding bias in the gradients. This is explained very well in the paper, and it is worth reading it to understand these issues.

