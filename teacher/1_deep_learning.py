# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deep Learning for Computer Vision
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" align="left" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>&nbsp;| Quentin Léturgie | Florient Chouteau | <a href="https://supaerodatascience.github.io/deep-learning/">https://supaerodatascience.github.io/deep-learning/</a>
#
# ## Session 1 : Deep Learning
#
# Welcome to this BE about applying images, convolutions and CNN. This is the second notebook of Deep Learning for Computer Vision
#
# 1.   **Generality about deep learning**
# 2.   Images, Convolutions and CNN
# 3.   CNN classifier
#
# It is recommended to use **Google Colab** to run these notebooks

# %%
# %matplotlib inline

# %% [markdown]
# ## **Session 1.1** : Pytorch
#
# This is heavily inspired from https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
#
# **What is PyTorch?**
#
# PyTorch is a Python-based scientific computing package serving two broad purposes:
#
#     A replacement for NumPy to use the power of GPUs and other accelerators.
#
#     An automatic differentiation library that is useful to implement neural networks.
#

# %% [markdown]
# ### 1.1.1 Tensors
#
# Tensors are a specialized data structure that are very similar to arrays
# and matrices. In PyTorch, we use tensors to encode the inputs and
# outputs of a model, as well as the model’s parameters.
#
# Tensors are similar to NumPy’s ndarrays, except that tensors can run on
# GPUs or other specialized hardware to accelerate computing. If you’re familiar with ndarrays, you’ll
# be right at home with the Tensor API. If not, follow along in this quick
# API walkthrough.
#

# %% jupyter={"outputs_hidden": false}
import numpy as np
import torch

# %% [markdown]
# #### Tensor Initialization
#
# Tensors can be initialized in various ways. Take a look at the following examples:
#
# **Directly from data**
#
# Tensors can be created directly from data. The data type is automatically inferred.
#
#

# %% jupyter={"outputs_hidden": false}
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# %% [markdown]
# **From a NumPy array**
#
# Tensors can be created from NumPy arrays (and vice versa - see `bridge-to-np-label`).
#
#

# %% jupyter={"outputs_hidden": false}
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# %% [markdown]
# **From another tensor:**
#
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
#
#

# %% jupyter={"outputs_hidden": false}
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# %% [markdown]
# **With random or constant values:**
#
# ``shape`` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
#
#

# %% jupyter={"outputs_hidden": false}
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %% [markdown]
# --------------
#
#
#

# %% [markdown]
# #### Tensor Attributes
#
# Tensor attributes describe their shape, datatype, and the device on which they are stored.
#
#

# %% jupyter={"outputs_hidden": false}
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %% [markdown]
# --------------
#
#
#

# %% [markdown]
# #### Tensor Operations
#
# Over 100 tensor operations, including transposing, indexing, slicing,
# mathematical operations, linear algebra, random sampling, and more are
# comprehensively described
# [here](https://pytorch.org/docs/stable/torch.html)_.
#
# Each of them can be run on the GPU (at typically higher speeds than on a
# CPU). If you’re using Colab, allocate a GPU by going to Edit > Notebook
# Settings.
#
#
#

# %% jupyter={"outputs_hidden": false}
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")

# %% [markdown]
# Try out some of the operations from the list.
# If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.
#
#
#

# %% [markdown]
# **Standard numpy-like indexing and slicing:**
#
#

# %% jupyter={"outputs_hidden": false}
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

# %% [markdown]
# **Joining tensors** You can use ``torch.cat`` to concatenate a sequence of tensors along a given dimension.
# See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)_,
# another tensor joining op that is subtly different from ``torch.cat``.
#
#

# %% jupyter={"outputs_hidden": false}
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# %% [markdown]
# **Multiplying tensors**
#
#

# %% jupyter={"outputs_hidden": false}
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# %% [markdown]
# This computes the matrix multiplication between two tensors
#
#

# %% jupyter={"outputs_hidden": false}
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# %% [markdown]
# **In-place operations**
# Operations that have a ``_`` suffix are in-place. For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.
#
#

# %% jupyter={"outputs_hidden": false}
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# %% [markdown]
# <div class="alert alert-info"><h4>Note</h4><p>In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
#      of history. Hence, their use is discouraged.</p></div>
#
#

# %% [markdown]
# --------------
#
#
#

# %% [markdown]
#
# #### Bridge with NumPy
#
# Tensors on the CPU and NumPy arrays can share their underlying memory
# locations, and changing one will change	the other.

# %% [markdown]
# #### Tensor to NumPy array
#
#

# %% jupyter={"outputs_hidden": false}
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# %% [markdown]
# A change in the tensor reflects in the NumPy array.
#
#

# %% jupyter={"outputs_hidden": false}
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# %% [markdown]
# #### NumPy array to Tensor
#
#

# %% jupyter={"outputs_hidden": false}
n = np.ones(5)
t = torch.from_numpy(n)

# %% [markdown]
# Changes in the NumPy array reflects in the tensor.
#
#

# %% jupyter={"outputs_hidden": false}
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# %% [markdown]
# ### 1.1.2 Backpropagation
#
# ``torch.autograd`` is PyTorch’s automatic differentiation engine that powers
# neural network training. In this section, you will get a conceptual
# understanding of how autograd helps a neural network train.
#
# #### Background
#
# Neural networks (NNs) are a collection of nested functions that are
# executed on some input data. These functions are defined by *parameters*
# (consisting of weights and biases), which in PyTorch are stored in
# tensors.
#
# Training a NN happens in two steps:
#
# **Forward Propagation**: In forward prop, the NN makes its best guess
# about the correct output. It runs the input data through each of its
# functions to make this guess.
#
# **Backward Propagation**: In backprop, the NN adjusts its parameters
# proportionate to the error in its guess. It does this by traversing
# backwards from the output, collecting the derivatives of the error with
# respect to the parameters of the functions (*gradients*), and optimizing
# the parameters using gradient descent. For a more detailed walkthrough
# of backprop, check out this [video from
# 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8)_.
#
#
# #### Usage in PyTorch
#
# Let's take a look at a single training step.
# For this example, we load a pretrained resnet18 model from ``torchvision``.
# We create a random data tensor to represent a single image with 3 channels, and height & width of 64,
# and its corresponding ``label`` initialized to some random values. Label in pretrained models has
# shape (1,1000).

# %% jupyter={"outputs_hidden": false}
import torch
from torchvision.models import ResNet18_Weights, resnet18

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# %% [markdown]
# Next, we run the input data through the model through each of its layers to make a prediction.
# This is the **forward pass**.
#
#
#

# %% jupyter={"outputs_hidden": false}
prediction = model(data)  # forward pass

# %% [markdown]
# We use the model's prediction and the corresponding label to calculate the error (``loss``).
# The next step is to backpropagate this error through the network.
# Backward propagation is kicked off when we call ``.backward()`` on the error tensor.
# Autograd then calculates and stores the gradients for each model parameter in the parameter's ``.grad`` attribute.
#
#
#

# %% jupyter={"outputs_hidden": false}
loss = (prediction - labels).sum()
loss.backward()  # backward pass

# %% [markdown]
# Next, we load an optimizer, in this case SGD with a learning rate of 0.01 and [momentum](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)_ of 0.9.
# We register all the parameters of the model in the optimizer.
#
#
#

# %% jupyter={"outputs_hidden": false}
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# %% [markdown]
# Finally, we call ``.step()`` to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in ``.grad``.
#
#
#

# %% jupyter={"outputs_hidden": false}
optim.step()  # gradient descent

# %% [markdown]
# At this point, you have everything you need to train your neural network.
# The below sections detail the workings of autograd - feel free to skip them.
#
#
#

# %% [markdown]
# --------------
#
#
#

# %% [markdown]
# #### Differentiation in Autograd
# Let's take a look at how ``autograd`` collects gradients. We create two tensors ``a`` and ``b`` with
# ``requires_grad=True``. This signals to ``autograd`` that every operation on them should be tracked.
#
#
#

# %% jupyter={"outputs_hidden": false}
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

# %% [markdown]
# We create another tensor ``Q`` from ``a`` and ``b``.
#
# \begin{align}Q = 3a^3 - b^2\end{align}
#
#

# %% jupyter={"outputs_hidden": false}
Q = 3 * a**3 - b**2

# %% [markdown]
# Let's assume ``a`` and ``b`` to be parameters of an NN, and ``Q``
# to be the error. In NN training, we want gradients of the error
# w.r.t. parameters, i.e.
#
# \begin{align}\frac{\partial Q}{\partial a} = 9a^2\end{align}
#
# \begin{align}\frac{\partial Q}{\partial b} = -2b\end{align}
#
#
# When we call ``.backward()`` on ``Q``, autograd calculates these gradients
# and stores them in the respective tensors' ``.grad`` attribute.
#
# We need to explicitly pass a ``gradient`` argument in ``Q.backward()`` because it is a vector.
# ``gradient`` is a tensor of the same shape as ``Q``, and it represents the
# gradient of Q w.r.t. itself, i.e.
#
# \begin{align}\frac{dQ}{dQ} = 1\end{align}
#
# Equivalently, we can also aggregate Q into a scalar and call backward implicitly, like ``Q.sum().backward()``.
#
#
#

# %% jupyter={"outputs_hidden": false}
external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)

# %% [markdown]
# Gradients are now deposited in ``a.grad`` and ``b.grad``
#
#

# %% jupyter={"outputs_hidden": false}
# check if collected gradients are correct
print(9 * a**2 == a.grad)
print(-2 * b == b.grad)

# %% [markdown]
# #### Optional Reading - Vector Calculus using ``autograd``
#
# Mathematically, if you have a vector valued function
# $\vec{y}=f(\vec{x})$, then the gradient of $\vec{y}$ with
# respect to $\vec{x}$ is a Jacobian matrix $J$:
#
# \begin{align}J
#      =
#       \left(\begin{array}{cc}
#       \frac{\partial \bf{y}}{\partial x_{1}} &
#       ... &
#       \frac{\partial \bf{y}}{\partial x_{n}}
#       \end{array}\right)
#      =
#      \left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)\end{align}
#
# Generally speaking, ``torch.autograd`` is an engine for computing
# vector-Jacobian product. That is, given any vector $\vec{v}$, compute the product
# $J^{T}\cdot \vec{v}$
#
# If $\vec{v}$ happens to be the gradient of a scalar function $l=g\left(\vec{y}\right)$:
#
# \begin{align}\vec{v}
#    =
#    \left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}\end{align}
#
# then by the chain rule, the vector-Jacobian product would be the
# gradient of $l$ with respect to $\vec{x}$:
#
# \begin{align}J^{T}\cdot \vec{v}=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)\left(\begin{array}{c}
#       \frac{\partial l}{\partial y_{1}}\\
#       \vdots\\
#       \frac{\partial l}{\partial y_{m}}
#       \end{array}\right)=\left(\begin{array}{c}
#       \frac{\partial l}{\partial x_{1}}\\
#       \vdots\\
#       \frac{\partial l}{\partial x_{n}}
#       \end{array}\right)\end{align}
#
# This characteristic of vector-Jacobian product is what we use in the above example;
# ``external_grad`` represents $\vec{v}$.
#
#
#

# %% [markdown]
# #### Computational Graph
#
# Conceptually, autograd keeps a record of data (tensors) & all executed
# operations (along with the resulting new tensors) in a directed acyclic
# graph (DAG) consisting of
# [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)_
# objects. In this DAG, leaves are the input tensors, roots are the output
# tensors. By tracing this graph from roots to leaves, you can
# automatically compute the gradients using the chain rule.
#
# In a forward pass, autograd does two things simultaneously:
#
# - run the requested operation to compute a resulting tensor, and
# - maintain the operation’s *gradient function* in the DAG.
#
# The backward pass kicks off when ``.backward()`` is called on the DAG
# root. ``autograd`` then:
#
# - computes the gradients from each ``.grad_fn``,
# - accumulates them in the respective tensor’s ``.grad`` attribute, and
# - using the chain rule, propagates all the way to the leaf tensors.
#
# Below is a visual representation of the DAG in our example. In the graph,
# the arrows are in the direction of the forward pass. The nodes represent the backward functions
# of each operation in the forward pass. The leaf nodes in blue represent our leaf tensors ``a`` and ``b``.
#
# ![](https://pytorch.org/tutorials/_images/dag_autograd.png)
#
# <div class="alert alert-info"><h4>Note</h4><p>**DAGs are dynamic in PyTorch**
#   An important thing to note is that the graph is recreated from scratch; after each
#   ``.backward()`` call, autograd starts populating a new graph. This is
#   exactly what allows you to use control flow statements in your model;
#   you can change the shape, size and operations at every iteration if
#   needed.</p></div>
#
# #### Exclusion from the DAG
#
# ``torch.autograd`` tracks operations on all tensors which have their
# ``requires_grad`` flag set to ``True``. For tensors that don’t require
# gradients, setting this attribute to ``False`` excludes it from the
# gradient computation DAG.
#
# The output tensor of an operation will require gradients even if only a
# single input tensor has ``requires_grad=True``.
#
#
#

# %% jupyter={"outputs_hidden": false}
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

# %% [markdown]
# In a NN, parameters that don't compute gradients are usually called **frozen parameters**.
# It is useful to "freeze" part of your model if you know in advance that you won't need the gradients of those parameters
# (this offers some performance benefits by reducing autograd computations).
#
# Another common usecase where exclusion from the DAG is important is for
# [finetuning a pretrained network](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)_
#
# In finetuning, we freeze most of the model and typically only modify the classifier layers to make predictions on new labels.
# Let's walk through a small example to demonstrate this. As before, we load a pretrained resnet18 model, and freeze all the parameters.
#
#

# %% jupyter={"outputs_hidden": false}
from torch import nn, optim

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# %% [markdown]
# Let's say we want to finetune the model on a new dataset with 10 labels.
# In resnet, the classifier is the last linear layer ``model.fc``.
# We can simply replace it with a new linear layer (unfrozen by default)
# that acts as our classifier.
#
#

# %% jupyter={"outputs_hidden": false}
model.fc = nn.Linear(512, 10)

# %% [markdown]
# Now all parameters in the model, except the parameters of ``model.fc``, are frozen.
# The only parameters that compute gradients are the weights and bias of ``model.fc``.
#
#

# %% jupyter={"outputs_hidden": false}
# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# %% [markdown]
# Notice although we register all the parameters in the optimizer,
# the only parameters that are computing gradients (and hence updated in gradient descent)
# are the weights and bias of the classifier.
#
# The same exclusionary functionality is available as a context manager in
# [torch.no_grad()](https://pytorch.org/docs/stable/generated/torch.no_grad.html)_
#

# %% [markdown]
# ## Session 1.2 : Neural Networks
#
# Neural networks can be constructed using the ``torch.nn`` package.
#
# Now that you had a glimpse of ``autograd``, ``nn`` depends on
# ``autograd`` to define models and differentiate them.
# An ``nn.Module`` contains layers, and a method ``forward(input)`` that
# returns the ``output``.
#
# For example, look at this network :
#
# ![](https://www.tibco.com/sites/tibco/files/media_entity/2021-05/neutral-network-diagram.svg)
#
# It is a simple feed-forward network. It takes the input, feeds it
# through several layers one after the other, and then finally gives the
# output.
#
# A typical training procedure for a neural network is as follows:
#
# - Define the neural network that has some learnable parameters (or
#   weights)
# - Iterate over a dataset of inputs
# - Process input through the network
# - Compute the loss (how far is the output from being correct)
# - Propagate gradients back into the network’s parameters
# - Update the weights of the network, typically using a simple update rule:
#   ``weight = weight - learning_rate * gradient``
#
# ### Define the network
#
# Let’s define this network:
#

# %% jupyter={"outputs_hidden": false}
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28, 64)  # 28x28 image => 784 inputs
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# %% [markdown]
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``
#
#

# %% jupyter={"outputs_hidden": false}
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# %% [markdown]
# Let's try a random 32x32 input.
# Note: expected input size of this net (LeNet) is 32x32. To use this net on
# the MNIST dataset, please resize the images from the dataset to 32x32.
#
#

# %% jupyter={"outputs_hidden": false}
input = torch.randn(1, 1, 28, 28)
out = net(input)
print(out)

# %% [markdown]
# Zero the gradient buffers of all parameters and backprops with random
# gradients:
#
#

# %% jupyter={"outputs_hidden": false}
net.zero_grad()
out.backward(torch.randn(1, 10))

# %% [markdown]
# ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Before proceeding further, let's recap all the classes you’ve seen so far.
#
# **Recap:**
#   -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
#      operations like ``backward()``. Also *holds the gradient* w.r.t. the
#      tensor.
#   -  ``nn.Module`` - Neural network module. *Convenient way of
#      encapsulating parameters*, with helpers for moving them to GPU,
#      exporting, loading, etc.
#   -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
#      registered as a parameter when assigned as an attribute to a*
#      ``Module``.
#   -  ``autograd.Function`` - Implements *forward and backward definitions
#      of an autograd operation*. Every ``Tensor`` operation creates at
#      least a single ``Function`` node that connects to functions that
#      created a ``Tensor`` and *encodes its history*.
#
# **At this point, we covered:**
#   -  Defining a neural network
#   -  Processing inputs and calling backward
#
# **Still Left:**
#   -  Computing the loss
#   -  Updating the weights of the network
#
# ### Loss Function
#
# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimates how far away the output is from the target.
#
# There are several different
# [loss functions](https://pytorch.org/docs/nn.html#loss-functions) under the
# nn package .
# A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# between the output and the target.
#
# For example:
#
#

# %% jupyter={"outputs_hidden": false}
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# %% [markdown]
# Now, if you follow ``loss`` in the backward direction, using its
# ``.grad_fn`` attribute, you will see a graph of computations that looks
# like this:
#
# ::
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> flatten -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# So, when we call ``loss.backward()``, the whole graph is differentiated
# w.r.t. the neural net parameters, and all Tensors in the graph that have
# ``requires_grad=True`` will have their ``.grad`` Tensor accumulated with the
# gradient.
#
# For illustration, let us follow a few steps backward:
#
#

# %% jupyter={"outputs_hidden": false}
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# %% [markdown]
# ### Backprop
#
# To backpropagate the error all we have to do is to ``loss.backward()``.
# You need to clear the existing gradients though, else gradients will be
# accumulated to existing gradients.
#
#
# Now we shall call ``loss.backward()``, and have a look at conv1's bias
# gradients before and after the backward.
#
#

# %% jupyter={"outputs_hidden": false}
net.zero_grad()  # zeroes the gradient buffers of all parameters

print("fc1.bias.grad before backward")
print(net.fc1.bias.grad)

loss.backward()

print("fc1.bias.grad after backward")
print(net.fc1.bias.grad)

# %% [markdown]
# Now, we have seen how to use loss functions.
#
# **Read Later:**
#
#   The neural network package contains various modules and loss functions
#   that form the building blocks of deep neural networks. A full list with
#   documentation is [here](https://pytorch.org/docs/nn).
#
# **The only thing left to learn is:**
#
#   - Updating the weights of the network
#
# ### Update the weights
# The simplest update rule used in practice is the Stochastic Gradient
# Descent (SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# We can implement this using simple Python code:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package: ``torch.optim`` that
# implements all these methods. Using it is very simple:
#
# First, let's save the weights : 

# %%
print("fc3 weights before update")
weights_before = np.copy(net.fc3.weight.data)
print(weights_before)

# %% jupyter={"outputs_hidden": false}
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update

# %% [markdown]
# Observe how gradient buffers had to be manually set to zero using
#   ``optimizer.zero_grad()``. This is because gradients are accumulated
#   as explained in the `Backprop`_ section.

# %%
print("fc3 weights after update")
weights_after = np.copy(net.fc3.weight.data)
print(weights_after)

# %%
print(weights_before - weights_after)

# %% [markdown]
# ## Session 1.3 : Your first training
#
#
# ![](https://fchouteau.github.io/isae-practical-deep-learning/static/img/pytorch_1.jpeg)
#
# ![](https://fchouteau.github.io/isae-practical-deep-learning/static/img/pytorch_2.jpeg)
#

# %% [markdown]
# This is it. You have seen how to define neural networks, compute loss and make
# updates to the weights of the network.
#
# Now you might be thinking,
#
# **What about data?**
#
# Generally, when you have to deal with image, text, audio or video data,
# you can use standard python packages that load data into a numpy array.
# Then you can convert this array into a ``torch.*Tensor``.
#
# -  For images, packages such as Pillow, OpenCV are useful
# -  For audio, packages such as scipy and librosa
# -  For text, either raw Python or Cython based loading, or NLTK and
#    SpaCy are useful
#
# Specifically for vision, we have created a package called
# ``torchvision``, that has data loaders for common datasets such as
# ImageNet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
# ``torchvision.datasets`` and ``torch.utils.data.DataLoader``.
#
# This provides a huge convenience and avoids writing boilerplate code.
#
# For this tutorial, we will use the MNIST dataset.
#
# **Training an digit classifier**
#
# We will do the following steps in order:
#
# 1. Load and normalize the MNIST training and test datasets
# 2. Define a  Neural Network
# 3. Define a loss function
# 4. Train the network on the training data
# 5. Test the network on the test data
#
# #### 1. Load and normalize MNIST
#
# Using ``torchvision``, it’s extremely easy to load MNIST.
#

# %% jupyter={"outputs_hidden": false}
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# %% [markdown]
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range

# %% jupyter={"outputs_hidden": false}
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

batch_size = 8

trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)

# %% [markdown]
# Let us show some of the training images, for fun.
#
#

# %% jupyter={"outputs_hidden": false}
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

# %% [markdown]
# #### 2. Define an ArtificialNeural Network
#
# Copy the neural network from the Neural Networks section before

# %% jupyter={"outputs_hidden": false}
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28, 64)  # 28x28 image => 784 inputs
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# %% [markdown]
# ### 3. Define a Loss function and optimizer
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
#
#

# %% jupyter={"outputs_hidden": false}
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %% [markdown]
# ### 4. Train the network
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
#
#

# %% jupyter={"outputs_hidden": false}
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# %% [markdown]
# Let's quickly save our trained model:
#
#

# %% jupyter={"outputs_hidden": false}
PATH = "./mnist_net.pth"
torch.save(net.state_dict(), PATH)

# %% [markdown]
# See [here](https://pytorch.org/docs/stable/notes/serialization.html)
# for more details on saving PyTorch models.
#
# ### 5. Test the network on the test data
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.
#
#

# %% jupyter={"outputs_hidden": false}
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

# %% [markdown]
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):
#
#

# %% jupyter={"outputs_hidden": false}
net = Net()
net.load_state_dict(torch.load(PATH))

# %% [markdown]
# Okay, now let us see what the neural network thinks these examples above are:
#
#

# %% jupyter={"outputs_hidden": false}
outputs = net(images)

# %% [markdown]
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
#
#

# %% jupyter={"outputs_hidden": false}
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

# %% [markdown]
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.
#
#

# %% jupyter={"outputs_hidden": false}
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")

# %% [markdown]
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:
#
#

# %% jupyter={"outputs_hidden": false}
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

# %% [markdown]
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# ## Training on GPU
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:
#
#

# %% jupyter={"outputs_hidden": false}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# %% [markdown]
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# ```python
# net.to(device)
# ```
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ```python
# inputs, labels = data[0].to(device), data[1].to(device)
# ```
#
# Why don't I notice MASSIVE speedup compared to CPU? Because your network
# is really small.
#
# **Exercise:** Try increasing the width of your network, see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images

# %%
