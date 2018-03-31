import numpy as np
from utils import *
import random

"""
    Read the dataset of dinosaur names,
    create a list of unique characters (such as a-z),
    and compute the dataset and vocabulary size.
"""

data = open('dinos.txt', 'r').read()
data = data.lower().replace('\t', '')
chars = list(set(data))
print chars
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' %
      (data_size, vocab_size))
# The characters are a-z (26 characters) plus the "\n" (or newline character), which plays a role similar to the `<EOS>` (or "End of sentence")

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
print(ix_to_char)

"""
 1.2 - Overview of the model

The model has the following structure:

- Initialize parameters
- Run the optimization loop
    - Forward propagation to compute the loss function
    - Backward propagation to compute the gradients with respect to the loss function
    - Clip the gradients to avoid exploding gradients
    - Using the gradients, update your parameter with the gradient descent update rule.
- Return the learned parameters

At each time-step, the RNN tries to predict what is the next character given the previous characters.
"""

# ## 2 - Building blocks of the model
#
# In this part, you will build two important blocks of the overall model:
# - Gradient clipping: to avoid exploding gradients
# - Sampling: a technique used to generate characters
#
# You will then apply these two functions to build the model.

# ### 2.1 - Clipping the gradients in the optimization loop
#
# In this section you will implement the `clip` function that you will call inside of your optimization loop. Recall that your overall loop structure usually consists of a forward pass, a cost computation, a backward pass, and a parameter update. Before updating the parameters, you will perform gradient clipping when needed to make sure that your gradients are not "exploding," meaning taking on overly large values.
#
# In the exercise below, you will implement a function `clip` that takes in a dictionary of gradients and returns a clipped version of gradients if needed. There are different ways to clip gradients; we will use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to lie between some range [-N, N]. More generally, you will provide a `maxValue` (say 10). In this example, if any component of the gradient vector is greater than 10, it would be set to 10; and if any component of the gradient vector is less than -10, it would be set to -10. If it is between -10 and 10, it is left alone.


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum to mitigate exploding gradients.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''

    gradkeys = ['dWaa', 'dWax', 'dWya', 'db', 'dby']
    grads = [gradients[i] for i in gradkeys]

    for i in range(len(grads)):
        grads[i] = np.clip(grads[i],
                           a_min=-1 * maxValue,
                           a_max=maxValue)

    gradients = dict(zip(gradkeys, grads))
    return gradients

# ### 2.2 - Sampling
#
# Now assume that your model is trained. You would like to generate new text (characters). The process of generation is explained in the picture below:
#

# **Exercise**: Implement the `sample` function below to sample characters. You need to carry out 4 steps:
#
# - **Step 1**: Pass the network the first "dummy" input $x^{\langle 1 \rangle} = \vec{0}$ (the vector of zeros). This is the default input before we've generated any characters. We also set $a^{\langle 0 \rangle} = \vec{0}$
#
# - **Step 2**: Run one step of forward propagation to get $a^{\langle 1 \rangle}$ and $\hat{y}^{\langle 1 \rangle}$. Here are the equations:
#
# $$ a^{\langle t+1 \rangle} = \tanh(W_{ax}  x^{\langle t \rangle } + W_{aa} a^{\langle t \rangle } + b)\tag{1}$$
#
# $$ z^{\langle t + 1 \rangle } = W_{ya}  a^{\langle t + 1 \rangle } + b_y \tag{2}$$
#
# $$ \hat{y}^{\langle t+1 \rangle } = softmax(z^{\langle t + 1 \rangle })\tag{3}$$
#
# Note that $\hat{y}^{\langle t+1 \rangle }$ is a (softmax) probability vector (its entries are between 0 and 1 and sum to 1). $\hat{y}^{\langle t+1 \rangle}_i$ represents the probability that the character indexed by "i" is the next character.  We have provided a `softmax()` function that you can use.
#
# - **Step 3**: Carry out sampling: Pick the next character's index according to the probability distribution specified by $\hat{y}^{\langle t+1 \rangle }$. This means that if $\hat{y}^{\langle t+1 \rangle }_i = 0.16$, you will pick the index "i" with 16% probability. To implement it, you can use [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html).
#
# Here is an example of how to use `np.random.choice()`:
# ```python
# np.random.seed(0)
# p = np.array([0.1, 0.0, 0.7, 0.2])
# index = np.random.choice([0, 1, 2, 3], p = p.ravel())
# ```
# This means that you will pick the `index` according to the distribution:
# $P(index = 0) = 0.1, P(index = 1) = 0.0, P(index = 2) = 0.7, P(index = 3) = 0.2$.
#
# - **Step 4**: The last step to implement in `sample()` is to overwrite the variable `x`, which currently stores $x^{\langle t \rangle }$, with the value of $x^{\langle t + 1 \rangle }$. You will represent $x^{\langle t + 1 \rangle }$ by creating a one-hot vector corresponding to the character you've chosen as your prediction. You will then forward propagate $x^{\langle t + 1 \rangle }$ in Step 1 and keep repeating the process until you get a "\n" character, indicating you've reached the end of the dinosaur name.

# In[22]:


def sample_char(x, a_prev, parameters):
    gradkeys = ['Waa', 'Wax', 'Wya', 'b', 'by']
    Waa, Wax, Wya, b, by = [parameters[i] for i in gradkeys]
    a = np.tanh(Wax.dot(x) + Waa.dot(a_prev) + b)
    z = Wya.dot(a) + by
    y = softmax(z)

    # Append the index to "indices" and break if EOS
    # idx = np.argmax(y)
    idx = np.random.choice(np.arange(vocab_size), p=y.ravel())
    return idx, a


def sample(parameters, char_to_ix, x=None, a_prev=None):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    vocab_size = parameters['by'].shape[0]
    n_a = parameters['Waa'].shape[1]

    x = np.zeros((vocab_size, 1)) if x is None else x
    a_prev = np.zeros((n_a, 1)) if a_prev is None else a_prev

    indices = []    # list of indices of the characters to generate
    idx = -1        # Flag to detect a newline character

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']
    for i in range(50):
        idx, a = sample_char(x, a_prev, parameters)
        indices.append(idx)
        if idx == newline_character:
            break
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a

    if (i == 49):
        indices.append(char_to_ix['\n'])

    return indices


# ## 3 - Building the language model
#
# It is time to build the character-level language model for text generation.
#
#
# ### 3.1 - Gradient descent
#
# In this section you will implement a function performing one step of stochastic gradient descent (with clipped gradients). You will go through the training examples one at a time, so the optimization algorithm will be stochastic gradient descent. As a reminder, here are the steps of a common optimization loop for an RNN:
#
# - Forward propagate through the RNN to compute the loss
# - Backward propagate through time to compute the gradients of the loss with respect to the parameters
# - Clip the gradients if necessary
# - Update your parameters using gradient descent
#
# **Exercise**: Implement this optimization process (one step of stochastic gradient descent).
#
# We provide you with the following functions:
#
# ```python
# def rnn_forward(X, Y, a_prev, parameters):
#     """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
#     It returns the loss' value as well as a "cache" storing values to be used in the backpropagation."""
#     ....
#     return loss, cache
#
# def rnn_backward(X, Y, parameters, cache):
#     """ Performs the backward propagation through time to compute the gradients of the loss with respect
#     to the parameters. It returns also all the hidden states."""
#     ...
#     return gradients, a
#
# def update_parameters(parameters, gradients, learning_rate):
#     """ Updates parameters using the Gradient Descent Update Rule."""
#     ...
#     return parameters
# ```

# GRADED FUNCTION: optimize

def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)

    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


# ### 3.2 - Training the model

# Given the dataset of dinosaur names, we use each line of the dataset (one name) as one training example. Every 100 steps of stochastic gradient descent, you will sample 10 randomly chosen names to see how the algorithm is doing. Remember to shuffle the dataset, so that stochastic gradient descent visits the examples in random order.
#
# **Exercise**: Follow the instructions and implement `model()`. When `examples[index]` contains one dinosaur name (string), to create an example (X, Y), you can use this:
# ```python
#         index = j % len(examples)
#         X = [None] + [char_to_ix[ch] for ch in examples[index]]
#         Y = X[1:] + [char_to_ix["\n"]]
# ```
# Note that we use: `index= j % len(examples)`, where `j = 1....num_iterations`, to make sure that `examples[index]` is always a valid statement (`index` is smaller than `len(examples)`).
# The first entry of `X` being `None` will be interpreted by `rnn_forward()` as setting $x^{\langle 0 \rangle} = \vec{0}$. Further, this ensures that `Y` is equal to `X` but shifted one step to the left, and with an additional "\n" appended to signify the end of the dinosaur name.

# In[40]:


# GRADED FUNCTION: model

def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        ### START CODE HERE ###

        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(examples)

        index = 4
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(
            X, Y, a_prev, parameters, learning_rate=0.01)

        ### END CODE HERE ###

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):

                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix)
                print_sample(sampled_indices, ix_to_char)

                # To get the same result for grading purposed, increment the seed by one.
                seed += 1

            print('\n')

    return parameters


# Run the following cell, you should observe your model outputting random-looking characters at the first iteration. After a few thousand iterations, your model should learn to generate reasonable-looking names.

# In[41]:

def predict(init, parameters):
    vocab_size = parameters['by'].shape[0]
    n_a = parameters['Waa'].shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []    # list of indices of the characters to generate
    idx = -1        # Flag to detect a newline character
    iter_var = 0
    x[char_to_ix[init[iter_var]]] = 1
    indices = [char_to_ix[init[iter_var]]]
    newline_character = char_to_ix['\n']

    for i in range(50):
        idx, a = sample_char(x, a_prev, parameters)
        if idx == newline_character:
            break
        x = np.zeros((vocab_size, 1))
        iter_var += 1
        if iter_var < len(init):
            x[char_to_ix[init[iter_var]]] = 1
            indices.append(char_to_ix[init[iter_var]])
        else:
            x[idx] = 1
            indices.append(idx)
        a_prev = a

    if (i == 49):
        indices.append(char_to_ix['\n'])

    return indices


parameters = model(data, ix_to_char, char_to_ix)
parameters
print_sample(predict('deb', parameters), ix_to_char)
