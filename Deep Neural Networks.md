# Key concepts on Deep Neural Networks

+ What is the "cache" used for in our implementation of forward propagation and backward propagation?
> We use it to pass variables computed during forward propagation to the corresponding backward propagation step. 
It contains useful values for backward propagation to compute derivatives. The "cache" records values from the forward propagation units and sends it to the backward propagation units because it is needed to compute the chain rule derivatives.

+ The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.

+ Vectorization allows you to compute forward propagation in an L-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?
> False. Note: We cannot avoid the for-loop iteration over the computations among layers.

+ (i) To compute the function using a shallow network circuit, you will need a large network (where we measure size by the number of logic gates in the network), but (ii) To compute it using a deep network circuit, you need only an exponentially smaller network.

