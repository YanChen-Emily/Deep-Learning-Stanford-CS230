# Practical Aspects about deep learning

+ Choice of learning rate

> **Reminder:** In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate $\alpha$ determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

> Different learning rates give different costs and thus different predictions results.
If the learning rate is too large, the cost may oscillate up and down. It may even diverge.
A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
In deep learning, we usually recommend that you:
Choose the learning rate that better minimizes the cost function.
If your model overfits, use other techniques to reduce overfitting. 

+ If you have 10,000,000 examples, how would you split the train/dev/test set?
> 98% train . 1% dev . 1% test

+ The dev and test set should come from the same distribution.

+ If your Neural Network model seems to have high variance, you may add regularization or get more training data.

+ What is weight decay: A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.

+ What happens when you increase the regularization hyperparameter lambda?
> Weights are pushed toward becoming smaller (closer to 0)

+ With the inverted dropout technique, at test time, You do not apply dropout (do not randomly eliminate units) and do not keep the 1/keep_prob factor in the calculations used in training.

+ Increasing the parameter keep_prob from (say) 0.5 to 0.6 will likely cause the following: (Check the two that apply)
> Reducing the regularization effect
Causing the neural network to end up with a lower training set error

+ Which of these techniques are useful for reducing variance (reducing overfitting)? 
> Dropout, L2 regularization, Data augmentation

+ Why do we normalize the inputs x?
> It makes the cost function faster to optimize.
