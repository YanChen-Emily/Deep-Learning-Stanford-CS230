# Practical aspects about deep learning

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

+ Train/dev/test sets workfolow 

You keep training your model on the training set, and use the development set or the holdout cross-validation set to see which model performs best on your dev set; and after having done this long enough, when you have a final model you want to evaluate, you can evaluate your final model on the test set to get an unbiased estimator of how your algorithm is doing.

In machine learning, you usually split your data like 70/30 or 60/20/20 with 1000, 10000 samples in total. 
In modern deep learning era, we sometimes/usually would have 1 million samples in total. And remember that development set is just used to evaluate which algorithm performs the best. So the dev set just needs to be big enough for you to test, say among the two algorithms or the ten algorithms which one is doing better. You may not need 20% of the data for that.

When you have a 1-million example, you may just split the data as 98/1/1.

+ Mismatched train/test distribution:
Training set: Cat pictures from webpages; Dev/test sets: Cat pictures from users using your app.

**Rule of Thumb: Make sure dev and test set come from the same distribution.**

### Initialization:

+ In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with $n^{[l]}=1$ for every layer, and the network is no more powerful than a linear classifier such as logistic regression. 

- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm. 
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.

<font color='blue'>
 
- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations. 

### L2-regularization

**Observations**:
- The value of $\lambda$ is a hyperparameter that you can tune using a dev set.
- L2 regularization makes your decision boundary smoother. If $\lambda$ is too large, it is also possible to "oversmooth", resulting in a model with high bias.

What is L2-regularization actually doing?

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes. 

**Dropout:**
- Dropout is a regularization technique.
- You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
- Apply dropout both during forward and backward propagation.
- During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. 

**Gradient Checking:**

$$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} \tag{2}$$

If this difference is small (say less than  $10^{-7}$ ), you can be quite confident that you have computed your gradient correctly. Otherwise, there may be a mistake in the gradient computation.

- Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).

- Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process. 

### SGD (stochastic gradient descent)

Stochastic Gradient Descent (SGD) is equivalent to mini-batch gradient descent where each mini-batch has just 1 example. The update rule that you have just implemented does not change. What changes is that you would be computing gradients on just one training example at a time, rather than on the whole training set. 

### Mini-Batch Gradient descent:
- Why is the best mini-batch size usually not 1 and not m, but instead something in-between?-
  - If the mini-batch size is 1, you lose the benefits of vectorization across examples in the mini-batch.
  - If the mini-batch size is m, you end up with batch gradient descent, which has to process the whole training set before making progress.
  
### Regularization

- You should not be using smaller networks because you are afraid of overtting. Instead, you should use as big of a neural network as your computational budget allows, and use other regularization techniques to control overtting. 
- Because bigger NNs have a greater representation power. And overfitting problems can be well addressed by proper regularization methods.

### Data Preprocessing
Normalization refers to normalizing the data dimensions so that they are of approximately the same scale. There are two common ways of achieving this normalization. One is to divide each dimension by its standard deviation, once it has been zero-centered: ```python (X /= np.std(X, axis = 0))```. 

Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. **It only makes sense to apply this preprocessing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm.**

In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.

#### PCA and Whitening

```python
# Assume input data matrix X of size [N x D]
X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
```

