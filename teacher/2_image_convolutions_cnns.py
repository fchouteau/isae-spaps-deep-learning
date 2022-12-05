# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="ab8a63a8"
# # Deep Learning for Computer Vision
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" align="left" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>&nbsp;| Quentin Léturgie | Florient Chouteau | <a href="https://supaerodatascience.github.io/deep-learning/">https://supaerodatascience.github.io/deep-learning/</a>
#
# ## Session 2 : Images, Convolutions and CNN
#
# Welcome to this BE about applying images, convolutions and CNN. This is the second notebook of Deep Learning for Computer Vision
#
# 1.   Generality about deep learning
# 2.   **Images, Convolutions and CNN**
# 3.   CNN classifier
#
# It is recommended to use **Google Colab** to run these notebooks

# %% [markdown] id="ANrvYsU8_HjY"
# ## **Session 2.1** : About images

# %% [markdown] id="946785cc"
# A digital image is an image composed of picture elements, also known as pixels, each with finite, discrete quantities of numeric representation for its intensity or gray level that is an output from its two-dimensional functions fed as input by its spatial coordinates denoted with x, y on the x-axis and y-axis, respectively.
#
# We represent images as matrixes,
#
# Images are made of pixels, and pixels are made of combinations of primary colors (in our case Red, Green and Blue). In this context, images have chanels that are the grayscale image of the same size as a color image, made of just one of these primary colors. For instance, an image from a standard digital camera will have a red, green and blue channel. A grayscale image has just one channel.
#
# In remote sensing, channels are often referred to as raster bands.
#
# <img src="https://miro.medium.com/max/1400/1*icINeO4H7UKe3NlU1fXqlA.jpeg" alt="drawing" width="400"/>
#
# For the rest of this workshop we will use the following axis conventions for images
#
# ![conventions](https://storage.googleapis.com/fchouteau-isae-deep-learning/static/image_coordinates.png)
#
# The reference library in python for working with images is https://scikit-image.org/
#
# We will just do basic image manipulation, but you [can look at all the examples](https://scikit-image.org/docs/stable/auto_examples/) if you need to get a better grasp of image processing

# %% id="0e379e97"
### LET'S CODE ###

# import some usefull library
import numpy as np
import skimage
import skimage.data
from matplotlib import pyplot as plt

# %% id="4ada1b9e" colab={"base_uri": "https://localhost:8080/", "height": 323} outputId="8dc98938-ff32-42d8-ff33-5c35a428a9dc"
# Load an image from skimage library
my_image = skimage.data.astronaut()

# Display the astronaut image
plt.figure(figsize=(5, 5))
plt.imshow(my_image)
plt.show()

# %% [markdown] id="b870a67f"
# * What is the height, width and number of channels of this image ?
# * In which order is the data represented ? Which dimensions are channels in ?
# * What is the image "dtype" ?
#
# How to get those informations?

# %% id="2ec3ccde" colab={"base_uri": "https://localhost:8080/"} outputId="4a356fc5-80d7-4d31-a015-19b2336442f8"
# Get image shape
img_height, img_width, img_channel = my_image.shape

print("height = ", img_height, "pixels")
print("width = ", img_width, "pixels")
print("Number of channels : ", img_channel)
print("Image data type : ", my_image.dtype)

# %% [markdown] id="878710a2"
# **uint8** means **unsigned interger on 8 bits**. This means that our image pixel values are integer between **0** and **255**.

# %% [markdown] id="d4e251ea"
# ### Exercice 1 : Extract subpart of the image
# In the following code, replace `...` by your answer.

# %% id="550fdec3"
# Extract subpart of the image (similar to crop), center on astronaut face.
# Define pixel position where astronaut face starts and ends for .
i_start = ...  # Pixel position between 0-512 and lower than i_end
i_end = ...  # Pixel position between 0-512 and higher than i_start
j_start = ...  # Pixel position between 0-512 and lower than i_end
j_end = ...  # Pixel position between 0-512 and higher than j_start


# Extract subpart of the image and display it
sub_img = my_image[i_start:i_end, j_start:j_end]

plt.figure(figsize=(5, 5))
plt.imshow(sub_img)
plt.show()

# %% [markdown] id="51bf8232"
# ### *Solution : Exercice 1* 

# %% id="ca11237b" colab={"base_uri": "https://localhost:8080/", "height": 391} outputId="054d8637-975e-425c-80fa-c3a71e67af46"
# Extract subpart of the image (similar to crop), center on astronaut face.
# Define pixel position where astronaut face starts and ends.
i_start = 0  # Pixel position between 0-512 and lower than i_end
i_end = 200  # Pixel position between 0-512 and higher than i_start
j_start = 150  # Pixel position between 0-512 and lower than i_end
j_end = 300  # Pixel position between 0-512 and higher than j_start


# Extract subpart of the image and display it
sub_img = my_image[i_start:i_end, j_start:j_end]

plt.figure(figsize=(5, 5))
plt.imshow(sub_img)
plt.show()

# We can get subpart image shape information
sub_img_height, sub_img_width, sub_img_channel = sub_img.shape

print("height = ", sub_img_height, "pixels")
print("width = ", sub_img_width, "pixels")
print("Number of channels : ", sub_img_channel)
print("Image data type : ", my_image.dtype)

# %% [markdown] id="c36fff05"
# In classical image representation, we use the [RGB color model](https://en.wikipedia.org/wiki/RGB_color_model) where the image is represented by three R,G,B channels (in that order).
#
# Usually we also use 8bits color depth

# %% [markdown] id="f9e3f807"
# ### Exercice 2 : Play with channels

# %% id="414550d8" colab={"base_uri": "https://localhost:8080/", "height": 323} outputId="311f2f76-aaef-42f9-d610-66474a5d4f59"
# Delete one color ot the image. For instance, set the red channel pixels value to 0.
# Display the image and check colors (especially red)
channel_number_to_delete = ...

img_without_red = np.copy(my_image)
img_without_red[:, :, channel_number_to_delete] = 0

plt.figure(figsize=(5, 5))
plt.imshow(img_without_red)
plt.show()

# %% id="cd7d87e3"
# Do the same for green band
channel_number_to_delete = ...

img_without_green = np.copy(my_image)
img_without_green[:, :, channel_number_to_delete] = 0

plt.figure(figsize=(5, 5))
plt.imshow(img_without_green)
plt.show()

# %% id="2ec71b08"
# Do the same for blue band
channel_number_to_delete = ...

img_without_blue = np.copy(my_image)
img_without_blue[:, :, channel_number_to_delete] = 0

plt.figure(figsize=(5, 5))
plt.imshow(img_without_blue)
plt.show()

# %% [markdown] id="0ca599fc"
# ### *Solution : Exercice 2*

# %% id="eeaf24cd" colab={"base_uri": "https://localhost:8080/", "height": 935} outputId="47881f36-5977-4bef-b11a-0926b885320d"
# Delete red band
channel_number_to_delete = 0

img_without_red = np.copy(my_image)
img_without_red[:, :, channel_number_to_delete] = 0

plt.figure(figsize=(5, 5))
plt.imshow(img_without_red)
plt.show()

# Delete green band
channel_number_to_delete = 1

img_without_green = np.copy(my_image)
img_without_green[:, :, channel_number_to_delete] = 0

plt.figure(figsize=(5, 5))
plt.imshow(img_without_green)
plt.show()

# Delete green band
channel_number_to_delete = 2

img_without_blue = np.copy(my_image)
img_without_blue[:, :, channel_number_to_delete] = 0

plt.figure(figsize=(5, 5))
plt.imshow(img_without_blue)
plt.show()


# %% [markdown] id="8e93aea7"
# ## **Session 2.2 :** Convolution

# %% [markdown] id="81a0309c"
# Someone may have told you that CNNs were the "thing" that made deep learning for image processing possible. But what are convolutions ?
#
# First, remember that you [learnt about convolutions a long time ago 😱](https://fr.wikipedia.org/wiki/Produit_de_convolution)
#
# <img src="https://betterexplained.com/ColorizedMath/img/Convolution.png" alt="drawing" width="400"/>
#
# So basically, we slide a filter over the signal. In 2D, this means
#
# <img src="https://miro.medium.com/max/535/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif" alt="drawing" width="400"/>
#
# One thing you can notice is that if we slide a filter over an image we "lose" pixels at the border. This is actually quite easy to compute : assuming a of size `2*k +1` we loose `k` pixels on each side of the image in each direction.
#
# ![](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/no_padding_no_strides.gif)
#
# If you want to get them back you have to "pad" (add values at the border, for examples zeroes) the image
#
# ![](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/arbitrary_padding_no_strides.gif)
#
# For more information, this website is excellent : https://cs231n.github.io/convolutional-networks/#conv
#
# Let's play with convolutions a little bit before actually doing CNNs.

# %% [markdown] id="8aadfdde"
# ### 2D Convolution without "depth"
#
# First, let's look at basic filtering over grayscale (1 channel) images. We will slide a filter over H,W spatial dimensions and get the result
#
# First, the convolution implementation without depth function is : 

# %% id="9545d023"
def convolve(img: np.array, kernel: np.array) -> np.array:
    """Apply a convolutionel kernel k on image img and return convolved image"""
    k = kernel.shape[0]
    h, w = img.shape[:2]
    p = int(k // 2)

    # Build output array : 2D array of zeros
    kernel = kernel.astype(np.float32)
    img = img.astype(np.float32)
    convolved_img = np.zeros(shape=(h - 2 * p, w - 2 * p)).astype(np.float32)

    # Iterate over the rows
    for i in range(h - 2 * p):
        # Iterate over the columns
        for j in range(w - 2 * p):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i : i + k, j : j + k]

            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    convolved_img = convolved_img.clip(0.0, 255.0).astype(np.uint8)

    return convolved_img


# %% [markdown] id="85fec939"
# Let's try some filters.
#
# ### Filter 1
# What happens if I use this filter as input ?
#
# ![identity](https://wikimedia.org/api/rest_v1/media/math/render/svg/1fbc763a0af339e3a3ff20af60a8a993c53086a7)

# %% id="e46ba213"
# Build convolution kernel
k = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
k = np.asarray(k)

print(k)

# %% id="ce967c10"
# Load image with only first channel
img = skimage.data.astronaut()
img = img[:, :, 0]

# Apply convolution kernel on image
convolved_img = convolve(img, k)

# Compare shapes
print("Shape before convolution (height, width) : ", img.shape)
print("Shape after convolution (height, width) : ", convolved_img.shape)

# Compare images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Before convolution")
ax[0].imshow(img, cmap="gray")
ax[1].set_title("After convolution")
ax[1].imshow(convolved_img, cmap="gray")
plt.show()

# %% [markdown] id="b74dffd1"
# What does this filter do? Nothing... It is identity filter.
#
# Note the loss of 2 pixels height and 2 pixels width... If we wanted to alleviate it we could do something like padding

# %% [markdown] id="82aea72b"
# ### Filter 2
# Too easy ! Let's try another filter
#
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/91256bfeece3344f8602e288d445e6422c8b8a1c)
#
# What does it do ? Take a guess

# %% id="d22ef50d"
k = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32)
k = k / k.sum()
print(k)

# %% id="811f0e73"
# Load image with only first channel
img = skimage.data.astronaut()
img = img[:, :, 0]

# Apply convolution kernel on image
convolved_img = convolve(img, k)

# Compare shapes
print("Shape before convolution (height, width) : ", img.shape)
print("Shape after convolution (height, width) : ", convolved_img.shape)

# Compare images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Before convolution")
ax[0].imshow(img, cmap="gray")
ax[1].set_title("After convolution")
ax[1].imshow(convolved_img, cmap="gray")
plt.show()

# %% [markdown] id="10ad88be"
# What does this filter do? Blurring

# %% [markdown] id="1ff7ca22"
# ### Filter 3
# Two more filters for edge detection (Sobel filter)
#
# ![image-2.png](attachment:image-2.png)

# %% id="9e78351a"
k_h = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

k_v = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

k_h = np.asarray(k_h)
k_v = np.asarray(k_v)

print("k_h :")
print(k_h)
print()
print("k_v :")
print(k_v)

# %% id="a7fd79e2"
# Load image with only first channel
img = skimage.data.astronaut()
img = img[:, :, 0]

# Apply convolution kernel on image
convolved_h_img = convolve(img, k_h)
convolved_v_img = convolve(img, k_v)

# Compare shapes
print("Shape before convolution (height, width) : ", img.shape)
print("Shape after convolution k_h (height, width) : ", convolved_h_img.shape)
print("Shape after convolution k_v (height, width) : ", convolved_v_img.shape)

# Compare images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].set_title("Before convolution")
ax[0].imshow(img, cmap="gray")
ax[1].set_title("After convolution by k_h")
ax[1].imshow(convolved_h_img, cmap="gray")
ax[2].set_title("After convolution by v_h")
ax[2].imshow(convolved_v_img, cmap="gray")
plt.show()

print("Sum those 2 results : ")
plt.figure()
plt.title("Sum of convolution by k_h and convolution by k_v")
plt.imshow(convolved_h_img + convolved_v_img, cmap="gray")
plt.show()

# %% [markdown] id="4d7d8779"
# What does this filter do? Edge detection 

# %% [markdown] id="618de390"
# If we wanted, we could learn the filters in order to do... cat classification, plane classification !
#
# There are many more filters that have been designed to do interesting things, you can find an interesting list here : https://en.wikipedia.org/wiki/Kernel_(image_processing)
#
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2141203%2F99dba888571cd6284b9b59903061aaa4%2Fko001.png?generation=1591783791920610&alt=media)
#
# **Takeaway message** : Kernel filtering (convolution) takes its root from classical image processing !

# %% [markdown] id="36ff73a6"
# ### Convolutions with depth
#
# <img src="https://i.stack.imgur.com/FjvuN.gif" alt="drawing" width="400"/>
#
# This is a convolution operator. It's the same as above, except our filter takes all channels of the image as input. So basically a "Convolution" layer is a filter

# %% [markdown] id="95568999"
# **Important** : In classical image processing, we use the (height, width, channels) convention, however in torch we prefer using (channels, height, width) convention 

# %% id="ff031e9a"
# Load image with 3 channels
img = skimage.data.astronaut()
img.shape

# %% id="292f18b6"
# To transpose an image, we use
img = img.transpose((2, 0, 1))  # change channel order
img.shape


# %% id="4a1717fa"
# This is the general implementation of convolutions
# It is the same function as function 'convolve' defined above but for multiple channel
def forward_convolution(conv_W, conv_b, data):
    """
    Compute the output from a convolutional layer given the weights and data.

    conv_W is of the shape (# output channels, # input channels, convolution width, convolution height )
    output_channels is the number of filters in the convolution

    conv_b is of the shape (# output channels)

    data is of the shape (# input channels, width, height)

    The output should be the result of a convolution and should be of the size:
        (# output channels, width - convolution width + 1, height -  convolution height + 1)

    Returns:
        The output of the convolution as a numpy array
    """

    conv_channels, _, conv_width, conv_height = conv_W.shape

    input_channels, input_width, input_height = data.shape

    output = np.zeros(
        (conv_channels, input_width - conv_width + 1, input_height - conv_height + 1)
    )

    for x in range(input_width - conv_width + 1):
        for y in range(input_height - conv_height + 1):
            for output_channel in range(conv_channels):
                output[output_channel, x, y] = (
                    np.sum(
                        np.multiply(
                            data[:, x : (x + conv_width), y : (y + conv_height)],
                            conv_W[output_channel, :, :, :],
                        )
                    )
                    + conv_b[output_channel]
                )

    return output


# %% tags=[] id="b0b85d17"
# We define random convolution kernel (weights) and random biases
weights = np.random.random((1, 3, 3, 3))
biases = np.random.random((3,))

# %% tags=[] id="82c32662"
# Convolve the input with the weights and bias
convolved_img = forward_convolution(weights, biases, img)


# %% id="4027bb79"
# Compare shapes
print("Shape before convolution (number of channels, height, width) : ", img.shape)
print(
    "Shape after convolution (number of channel, height, width) : ", convolved_img.shape
)

# Compare images (Don't forget that matplotlib uses (h,w,c) to plot images !)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title("Before convolution")
ax[0].imshow(img.transpose((1, 2, 0)))
ax[1].set_title("After convolution")
ax[1].imshow(convolved_img[0], cmap="gray")
plt.show()

# Print filter and biases values
print(f"Filter:\n {weights}")
print()
print("Bias:", biases)

# %% [markdown] id="27e59773"
# What does this filter do? An unknown random operation

# %% [markdown] id="54c1ded4"
# Some useful resources for more information :
#
# - The DL class https://github.com/fchouteau/deep-learning/blob/main/deep/Deep%20Learning.ipynb
# - https://github.com/vdumoulin/conv_arithmetic
# - https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1

# %% [markdown] id="b79ec18d"
# ## **Session 2.3** : Convolution Neural Network

# %% [markdown] id="d39b3fa5"
#
#
# ```
# # Ce texte est au format code
# ```
#
# I shamelessly copy pasted code from this excellent class : https://github.com/Atcold/pytorch-Deep-Learning/blob/master/06-convnet.ipynb
#
# Remember, an Artificial Neural Network is a stack of 
#
# - "Fully Connected" layers
# - Non linearities
#
# A Convolutional Neural Network is a stack of
# - Convolutional Layers aka Filter Banks
#     - Increase dimensionality
#     - Projection on overcomplete basis
#     - Edge detections
# - Non-linearities
#     - Sparsification
#     - Typically Rectified Linear Unit (ReLU): ReLU(x)=max⁡(x,0)\text{ReLU}(x) = \max(x, 0)ReLU(x)=max(x,0)
# - Pooling
#     - Aggregating over a feature map
#     - Example : Maximum
#
# ![](https://cdn-media-1.freecodecamp.org/images/Dgy6hBvOvAWofkrDM8BclOU3E3C2hqb25qBb)
#
# <img src="https://production-media.paperswithcode.com/methods/MaxpoolSample2.png" alt="drawing" width="400"/>
#
# Max pooling operations

# %% [markdown] id="a7db0840"
# Why do CNNs works ?
#
# To perform well, we need to incorporate some prior knowledge about the problem
#
#     Assumptions helps us when they are true
#     They hurt us when they are not
#     We want to make just the right amount of assumptions, not more than that
#     
# In Deep Learning
#
#     Many layers: compositionality
#     Convolutions: locality + stationarity of images
#     Pooling: Invariance of object class to translations

# %% [markdown] id="iuqVb4bS2lTP"
# In pytorch many function exists to define layers easily and are stored in `torch.nn` package :
#
#
# *   `nn.Conv2d` : Build 2 dimension convolutional layer
# *   `nn.ReLU` : Build Rectified Linear Unit layer
# *   `nn.MaxPool2d` : Build 2 dimension max pooling layer
#
#
# To see all available layers : https://pytorch.org/docs/stable/nn.html
#
#

# %% id="KMV96Yh42RIG"
### LET'S CODE ###
# We will build our first cnn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# %% [markdown] id="gofB8nTF9AVU"
# CNN MODEL

# %% id="leXC3D1R2hcY"
# We define a CNN through a python class
class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()

    def forward(self, x):
        x = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5)(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)

        x = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)

        x = nn.Flatten()(x)
        x = nn.Linear(24 * 4 * 4, 50)(x)
        x = nn.ReLU()(x)

        x = nn.Linear(50, 10)(x)
        x = nn.LogSoftmax(dim=1)(x)

        return x


cnn_model = CNN_Network()

# %% [markdown] id="is_k8NsZ8YzC"
# DATASET

# %% colab={"base_uri": "https://localhost:8080/", "height": 383} id="YKBLuDRV5SXj" outputId="1b049df0-b8d1-4044-b44e-ce8a63f258a5"
# To build our model we need to define input and output shapes
# For this example we use basic dataset : mnist. We load it thanks to PyTorch
mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Transform image to tensor
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)  # Normalize each sample by global mean/std

mnist_train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=mnist_transform
)
mnist_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=mnist_transform
)


# Print some stats
print("Number of samples in MNIST train dataset : ", len(mnist_train_dataset))
print("Number of samples in MNIST test dataset : ", len(mnist_test_dataset))
print("Dataset image shape : ", np.asarray(mnist_train_dataset[0][0]).shape)
print()

# To visualize this train dataset, we plot few samples
n = 10
print(n, "samples of MNIST train dataset")

print("train labels : ", end="")
plt.figure(figsize=(15, 5))
for i in range(n):
    print(mnist_train_dataset[i][1], end=", ")
    plt.subplot(1, n, i + 1)
    plt.imshow(mnist_train_dataset[i][0][0], cmap="gray")
plt.show()

print()
# To visualize this test dataset, we plot few samples
n = 10
print(n, "samples of MNIST test dataset")

print("test labels : ", end="")
plt.figure(figsize=(15, 5))
for i in range(n):
    print(mnist_test_dataset[i][1], end=", ")
    plt.subplot(1, n, i + 1)
    plt.imshow(mnist_test_dataset[i][0][0], cmap="gray")
plt.show()

# %% [markdown] id="NBLmPXwVPkPa"
# DATA LOADER

# %% id="3WuIfjiZPmDw"
input_size = 28 * 28  # images are 28x28 pixels
output_size = 10  # there are 10 classes

train_loader = torch.utils.data.DataLoader(
    mnist_train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    mnist_test_dataset, batch_size=1000, shuffle=True
)

# %% [markdown] id="10762497"
# Switching between CPU and GPU in PyTorch is controlled via a device string, which will seemlessly determine whether GPU is available, falling back to CPU if not:

# %% id="2a09379c"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown] id="P3brB4wrUA41"
# TRAINING AND TESTING STEPS

# %% id="A2xQGm1XVOV4"
accuracy_list = []


def train(epoch, model):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # send data and model to same device
        data, target = data.to(device), target.to(device)

        # Zero grad optimize
        optimizer.zero_grad()

        # Apply the model on data and get output prediction
        prediction = model(data)

        # Compute loss between prediction and target
        loss = F.nll_loss(prediction, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # send data and model to same device
        data, target = data.to(device), target.to(device)

        # Apply the model on data and get output prediction
        prediction = model(data)
        test_loss += F.nll_loss(
            prediction, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = prediction.data.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )


# %% colab={"base_uri": "https://localhost:8080/", "height": 382} id="UcXXkBnDW10I" outputId="3d04ec9b-3a03-4410-dea3-a39d00393cac"
model = CNN_Network()


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(0, 1):
    train(epoch, model)
    test(cnn_model)

# %% id="715b8f46"
accuracy_list = []


def train(epoch, model, perm=torch.arange(0, 784).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.to(device), target.to(device)

        # permute pixels
        data = data.view(-1, 28 * 28)
        data = data[:, perm]
        data = data.view(-1, 1, 28, 28)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # send to device
        data, target = data.to(device), target.to(device)

        # permute pixels
        data = data.view(-1, 28 * 28)
        data = data[:, perm]
        data = data.view(-1, 1, 28, 28)
        output = model(data)
        test_loss += F.nll_loss(
            output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )


# %% id="a723ed09"
# function to count number of parameters
def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


# Create two models: One ANN vs One CNN
class FullyConnected2Layers(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(FullyConnected2Layers, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.network(x)


class CNN(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_feature, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_feature, n_feature, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(n_feature * 4 * 4, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, verbose=False):
        return self.network(x)


# %% [markdown] id="70c2a4cf"
# ### Definitions

# %% id="4cfbe195"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# %% id="2dff475e"
input_size = 28 * 28  # images are 28x28 pixels
output_size = 10  # there are 10 classes

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=1000,
    shuffle=True,
)

# %% id="43644c21"
# show some images
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    image, _ = train_loader.dataset.__getitem__(i)
    plt.imshow(image.squeeze().numpy())
    plt.axis("off");

# %% [markdown] id="169507f0"
# ### CNNs vs Fully Connected Layers

# %% [markdown] id="f839a688"
# A small FullyConnected ANN

# %% id="6e6dc883"
n_hidden = 8  # number of hidden units

model_fnn = FullyConnected2Layers(input_size, n_hidden, output_size)
model_fnn.to(device)
optimizer = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5)

print("Number of parameters: {}".format(get_n_params(model_fnn)))

for epoch in range(0, 1):
    train(epoch, model_fnn)
    test(model_fnn)

# %% [markdown] id="4d9fe8fc"
# A CNN with the same number of parameters

# %% id="01a9716d"
# Training settings
n_features = 6  # number of feature maps

model_cnn = CNN(input_size, n_features, output_size)
model_cnn.to(device)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)

print("Number of parameters: {}".format(get_n_params(model_cnn)))

for epoch in range(0, 1):
    train(epoch, model_cnn)
    test(model_cnn)

# %% [markdown] id="b6dd397f"
# The ConvNet performs better with the same number of parameters, thanks to its use of prior knowledge about images
#
#     Use of convolution: Locality and stationarity in images
#     Pooling: builds in some translation invariance

# %% [markdown] id="51c93c84"
# ### What happens when CNNs assumptions are not true ?
#
# We will deterministically permute pixels so that the content of an image is respected but not its structure
#
# Basically transform some positions into others
#
# And we will train networks on this

# %% id="fa5d3bc5"
perm = torch.randperm(784)
plt.figure(figsize=(16, 12))
for i in range(10):
    image, _ = train_loader.dataset.__getitem__(i)
    # permute pixels
    image_perm = image.view(-1, 28 * 28).clone()
    image_perm = image_perm[:, perm]
    image_perm = image_perm.view(-1, 1, 28, 28)
    plt.subplot(4, 5, i + 1)
    plt.imshow(image.squeeze().numpy())
    plt.axis("off")
    plt.subplot(4, 5, i + 11)
    plt.imshow(image_perm.squeeze().numpy())
    plt.axis("off")

# %% id="37de1337"
# Training settings
n_features = 6  # number of feature maps

model_cnn = CNN(input_size, n_features, output_size)
model_cnn.to(device)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
print("Number of parameters: {}".format(get_n_params(model_cnn)))

for epoch in range(0, 1):
    train(epoch, model_cnn, perm)
    test(model_cnn, perm)

# %% id="6441b819"
n_hidden = 8  # number of hidden units

model_fnn = FullyConnected2Layers(input_size, n_hidden, output_size)
model_fnn.to(device)
optimizer = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5)
print("Number of parameters: {}".format(get_n_params(model_fnn)))

for epoch in range(0, 1):
    train(epoch, model_fnn, perm)
    test(model_fnn, perm)

# %% [markdown] id="11d0fb79"
# **Takeaway messages**
#
# The ConvNet's performance drops when we permute the pixels, but the Fully-Connected Network's performance stays the same
#
#     ConvNet makes the assumption that pixels lie on a grid and are stationary/local
#     It loses performance when this assumption is wrong
#     The fully-connected network does not make this assumption
#     It does less well when it is true, since it doesn't take advantage of this prior knowledge
#     But it doesn't suffer when the assumption is wrong

# %% id="8ce44a07"
plt.bar(
    ("NN normal", "CNN normal", "CNN scrambled", "NN scrambled"),
    accuracy_list,
    width=0.4,
)
plt.ylim((min(accuracy_list) - 5, 96))
plt.ylabel("Accuracy [%]")
for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(10)
plt.title("Performance comparison");
