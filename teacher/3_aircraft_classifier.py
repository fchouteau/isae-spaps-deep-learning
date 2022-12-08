# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# %% [markdown] {"id": "2iUXCk7tC1x5", "tags": []}
# # Session 3 : Training your first aircraft classifier with pytorch
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" align="left" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>&nbsp;| Florient Chouteau | Quentin Léturgie <a href="https://supaerodatascience.github.io/deep-learning/">https://supaerodatascience.github.io/deep-learning/</a>

# %% [markdown] {"id": "yfn1RtChC1yD", "tags": []}
# ## Intro
#
# The objectives of this session is to apply what you learned during the previous notebooks on Deep Learning on a real dataset of satellite images.
#
# Most of the vocabulary and concepts of Deep Learning and Convolutionnal Neural Network has been defined on the class linked above so you should refer to it.
#
# In this session you will:
# - Train a basic NN on a basic dataset
# - Plot ROC curve & confusion matrix to diagnose your dataset
#
# During session 2 you will be experimenting with harder datasets
#
# If you haven't done so, go to the previous notebooks to get a hands on pytorch and CNNs
#
#
# **First steps**
# - Activate the GPU runtime in colab
# - Check using `!nvidia-smi` that you detect it

# %%
# Bash command nvidia-smi prints the current GPU usage and process list if GPU is detected
# !nvidia-smi

# %%
# %matplotlib inline

# %%
# Put your imports here
import numpy as np

# %% [markdown] {"tags": []}
# ## Utils Definition / Ignore & hide
#
# Execute once and hide this section

# %%
# IGNORE THIS FUNCTION THIS IS AN UTIL, just execute once
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
UNKNOWN_SIZE = "?"


class LayerSummary:
    """Summary class for a single layer in a :class:`~torch.nn.Module`.

    It collects the following information:

    * Type of the layer (e.g. Linear, BatchNorm1d, ...)
    * Input shape
    * Output shape
    * Number of parameters

    The input and output shapes are only known after the example input array was
    passed through the model.

    Example::
        >>> model = torch.nn.Conv2d(3, 8, 3)
        >>> summary = LayerSummary(model)
        >>> summary.num_parameters
        224
        >>> summary.layer_type
        'Conv2d'
        >>> output = model(torch.rand(1, 3, 5, 5))
        >>> summary.in_size
        [1, 3, 5, 5]
        >>> summary.out_size
        [1, 8, 3, 3]

    Args:
        module: A module to summarize
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module
        self._hook_handle = self._register_hook()
        self._in_size: Optional[Union[str, List]] = None
        self._out_size: Optional[Union[str, List]] = None

    def __del__(self) -> None:
        self.detach_hook()

    def _register_hook(self) -> Optional[RemovableHandle]:
        """Registers a hook that computes the input/output size(s) on the first forward pass.

        If the hook is called, it will remove itself from the from the module, meaning that
        recursive models will only record their input- and output shapes once.

        Registering hooks on :class:`~torch.jit.ScriptModule` is not supported.

        Return:
            A handle for the installed hook, or ``None`` if registering the hook is not possible.
        """

        def hook(_: nn.Module, inp: Any, out: Any) -> None:
            if len(inp) == 1:
                inp = inp[0]
            self._in_size = parse_batch_shape(inp)
            self._out_size = parse_batch_shape(out)
            assert self._hook_handle is not None
            self._hook_handle.remove()

        handle = None
        if not isinstance(self._module, torch.jit.ScriptModule):
            handle = self._module.register_forward_hook(hook)
        return handle

    def detach_hook(self) -> None:
        """Removes the forward hook if it was not already removed in the forward pass.

        Will be called after the summary is created.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()

    @property
    def in_size(self) -> Union[str, List]:
        return self._in_size or UNKNOWN_SIZE

    @property
    def out_size(self) -> Union[str, List]:
        return self._out_size or UNKNOWN_SIZE

    @property
    def layer_type(self) -> str:
        """Returns the class name of the module."""
        return str(self._module.__class__.__name__)

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(
            np.prod(p.shape) if not _is_lazy_weight_tensor(p) else 0
            for p in self._module.parameters()
        )


class ModelSummary:
    """Generates a summary of all layers in a :class:`~torch.nn.Module`.

    Args:
        model: The model to summarize (also referred to as the root module).
        max_depth: Maximum depth of modules to show. Use -1 to show all modules or 0 to show no
            summary. Defaults to 1.
        example_input_array (torch.Tensor): If provided, and example input aray which will be used
            to infer the shape of tensors throughout the model.

    The string representation of this summary prints a table with columns containing
    the name, type and number of parameters for each layer.
    The root module may also have an attribute ``example_input_array`` as shown in the example
    below.

    If present, the root module will be called with it as input to determine the
    intermediate input- and output shapes of all layers. Supported are tensors and
    nested lists and tuples of tensors. All other types of inputs will be skipped and show as `?`
    in the summary table. The summary will also display `?` for layers not used in the forward pass.

    Example::
        >>> from torch.nn import Module
        >>> class LitModel(Module):
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512))
        ...         self.example_input_array = torch.zeros(10, 256)  # optional
        ...
        ...     def forward(self, x):
        ...         return self.net(x)
        ...
        >>> model = LitModel()
        >>> ModelSummary(model, max_depth=1)  # doctest: +NORMALIZE_WHITESPACE
          | Name | Type       | Params | In sizes  | Out sizes
        ------------------------------------------------------------
        0 | net  | Sequential | 132 K  | [10, 256] | [10, 512]
        ------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
        >>> ModelSummary(model, max_depth=-1)  # doctest: +NORMALIZE_WHITESPACE
          | Name  | Type        | Params | In sizes  | Out sizes
        --------------------------------------------------------------
        0 | net   | Sequential  | 132 K  | [10, 256] | [10, 512]
        1 | net.0 | Linear      | 131 K  | [10, 256] | [10, 512]
        2 | net.1 | BatchNorm1d | 1.0 K    | [10, 512] | [10, 512]
        --------------------------------------------------------------
        132 K     Trainable params
        0         Non-trainable params
        132 K     Total params
        0.530     Total estimated model params size (MB)
    """

    def __init__(self, model, max_depth=1, example_input_array=None) -> None:
        self._model = model

        # temporary mapping from mode to max_depth
        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or > 0, got {max_depth}.")

        self._max_depth = max_depth
        self._example_input_array = example_input_array
        self._layer_summary = self.summarize(example_input_array=example_input_array)
        # 1 byte -> 8 bits
        # TODO: how do we compute precision_megabytes in case of mixed precision?
        precision = 32  # Fixme: Get precision from global dtype attribute
        self._precision_megabytes = (precision / 8.0) * 1e-6

    @property
    def named_modules(self) -> List[Tuple[str, nn.Module]]:
        mods: List[Tuple[str, nn.Module]]
        if self._max_depth == 0:
            mods = []
        elif self._max_depth == 1:
            # the children are the top-level modules
            mods = list(self._model.named_children())
        else:
            mods = self._model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        return mods

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> List[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> List:
        return [layer.in_size for layer in self._layer_summary.values()]

    @property
    def out_sizes(self) -> List:
        return [layer.out_size for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> List[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def total_parameters(self) -> int:
        return sum(
            p.numel() if not _is_lazy_weight_tensor(p) else 0
            for p in self._model.parameters()
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(
            p.numel() if not _is_lazy_weight_tensor(p) else 0
            for p in self._model.parameters()
            if p.requires_grad
        )

    @property
    def model_size(self) -> float:
        # todo: seems it does not work with quantized models - it returns 0.0
        return self.total_parameters * self._precision_megabytes

    def summarize(self, example_input_array=None) -> Dict[str, LayerSummary]:
        summary = OrderedDict(
            (name, LayerSummary(module)) for name, module in self.named_modules
        )

        if example_input_array is not None:
            self._forward_example_input(example_input_array)

        for layer in summary.values():
            layer.detach_hook()

        if self._max_depth >= 1:
            # remove summary entries with depth > max_depth
            for k in [k for k in summary if k.count(".") >= self._max_depth]:
                del summary[k]

        return summary

    def _forward_example_input(self, example_input_array) -> None:
        """Run the example input through each layer to get input- and output sizes."""
        model = self._model

        mode = model.training
        model.eval()
        with torch.no_grad():
            # let the model hooks collect the input- and output shapes
            if isinstance(example_input_array, (list, tuple)):
                model(*example_input_array)
            elif isinstance(example_input_array, dict):
                model(**example_input_array)
            else:
                model(example_input_array)
        model.train(mode)  # restore mode of module

    def _get_summary_data(self) -> List[Tuple[str, List[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size
        """
        arrays = [
            (" ", list(map(str, range(len(self._layer_summary))))),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", list(map(get_human_readable_count, self.param_nums))),
        ]
        if self._example_input_array is not None:
            arrays.append(("In sizes", [str(x) for x in self.in_sizes]))
            arrays.append(("Out sizes", [str(x) for x in self.out_sizes]))

        return arrays

    def __str__(self) -> str:
        arrays = self._get_summary_data()

        total_parameters = self.total_parameters
        trainable_parameters = self.trainable_parameters
        model_size = self.model_size

        return _format_summary_table(
            total_parameters, trainable_parameters, model_size, *arrays
        )

    def __repr__(self) -> str:
        return str(self)


def parse_batch_shape(batch: Any) -> Union[str, List]:
    if hasattr(batch, "shape"):
        return list(batch.shape)

    if isinstance(batch, (list, tuple)):
        shape = [parse_batch_shape(el) for el in batch]
        return shape

    return UNKNOWN_SIZE


def _format_summary_table(
    total_parameters: int,
    trainable_parameters: int,
    model_size: float,
    *cols: Tuple[str, List[str]],
) -> str:
    """Takes in a number of arrays, each specifying a column in the summary table, and combines them all into one
    big string defining the summary table that are nicely formatted."""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width

    summary += "\n" + s.format(get_human_readable_count(trainable_parameters), 10)
    summary += "Trainable params"
    summary += "\n" + s.format(
        get_human_readable_count(total_parameters - trainable_parameters), 10
    )
    summary += "Non-trainable params"
    summary += "\n" + s.format(get_human_readable_count(total_parameters), 10)
    summary += "Total params"
    summary += "\n" + s.format(get_formatted_model_size(model_size), 10)
    summary += "Total estimated model params size (MB)"

    return summary


def get_formatted_model_size(total_model_size: float) -> str:
    return f"{total_model_size:,.3f}"


def get_human_readable_count(number: int) -> str:
    """Abbreviates an integer number with K, M, B, T.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


def _is_lazy_weight_tensor(p: Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate.",
            UserWarning,
        )
        return True

    return False


def summarize(module, max_depth=None, example_input_array=None) -> ModelSummary:
    """Summarize a PyTorch module.

    Args:
        module: The module to summarize.
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0
            turns the layer summary off. Default: 1.
        example_input_array (torch.Tensor): If provided, and example input aray which will be used
            to infer the shape of tensors throughout the model.

    Return:
        The model summary object
    """
    max_depth = 1 if max_depth is None else max_depth
    model_summary = ModelSummary(
        module, max_depth=max_depth, example_input_array=example_input_array
    )

    return model_summary


# %% [markdown]
# ## Dataset
#
# Collect and explore the dataset

# %%
# Configuration variables
TOY_DATASET_URL = "https://storage.googleapis.com/fchouteau-isae-deep-learning/large_aircraft_dataset.npz"

# %% [markdown]
# ### Image (reminders)
#
# A digital image is an image composed of picture elements, also known as pixels, each with finite, discrete quantities of numeric representation for its intensity or gray level that is an output from its two-dimensional functions fed as input by its spatial coordinates denoted with x, y on the x-axis and y-axis, respectively.
#
# We represent images as matrixes,
#
# Images are made of pixels, and pixels are made of combinations of primary colors (in our case Red, Green and Blue). In this context, images have chanels that are the grayscale image of the same size as a color image, made of just one of these primary colors. For instance, an image from a standard digital camera will have a red, green and blue channel. A grayscale image has just one channel.
#
# In geographic information systems, channels are often referred to as raster bands.
#
# ![img](https://static.packt-cdn.com/products/9781789613964/graphics/e91171a3-f7ea-411e-a3e1-6d3892b8e1e5.png)
#
#
# For the rest of this workshop we will use the following axis conventions for images
#
# ![conventions](https://storage.googleapis.com/fchouteau-isae-deep-learning/static/image_coordinates.png)

# %% [markdown]
# ### Downloading the dataset
#
# We will be using [numpy datasources](https://docs.scipy.org/doc/numpy/reference/generated/numpy.DataSource.html?highlight=datasources) to download the dataset. DataSources can be local files or remote files/URLs. The files may also be compressed or uncompressed. DataSource hides some of the low-level details of downloading the file, allowing you to simply pass in a valid file path (or URL) and obtain a file object.
#
# The dataset is in npz format which is a packaging format where we store several numpy arrays in key-value format
#
# Note:
# If you get an error with the code below run:
# ```python
# !gsutil -m cp -r gs://isae-deep-learning/toy_aircraft_dataset.npz /tmp/storage.googleapis.com/isae-deep-learning/toy_aircraft_dataset.npz
# ```
# in a cell above the cell below

# %%
ds = np.DataSource(destpath="/tmp/")
f = ds.open(TOY_DATASET_URL, "rb")

toy_dataset = np.load(f)
trainval_images = toy_dataset["train_images"]
trainval_labels = toy_dataset["train_labels"]
test_images = toy_dataset["test_images"]
test_labels = toy_dataset["test_labels"]

# %% [markdown]
# ### A bit of data exploration

# %% [markdown]
# **Q1. Labels counting**
#
# a. What is the dataset size ?
#
# b. How many images representing aircrafts ?
#
# c. How many images representing backgrounds ?
#
# d. What are the dimensions (height and width) of the images ? What are the number of channels ?

# %% [markdown]
# **Q2. Can you plot at least 8 examples of each label ? In a 4x4 grid ?**

# %% [markdown]
# Here are some examples that help you answer this question. Try them and make your own. A well-understandood dataset is the key to an efficient model.

# %%
import cv2
import matplotlib.pyplot as plt

# %%
LABEL_NAMES = ["Not an aircraft", "Aircraft"]

print("Labels counts :")
for l, c, label in zip(*np.unique(trainval_labels, return_counts=True), LABEL_NAMES):
    print(f" Label: {label} , value: {l}, count: {c}")

for l, label in enumerate(LABEL_NAMES):
    print(
        f"Examples shape for label {l} : {trainval_images[trainval_labels == l, ::].shape}"
    )

# %%
LABEL_NAMES = ["Not an aircraft", "Aircraft"]

print("Labels counts for test dataset :")
for l, c, label in zip(*np.unique(test_labels, return_counts=True), LABEL_NAMES):
    print(f" Label: {label} , value: {l}, count: {c}")

for l, label in enumerate(LABEL_NAMES):
    print(f"Examples shape for label {l} : {test_images[test_labels == l, ::].shape}")

# %%
grid_size = 4
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = np.copy(trainval_images[i * grid_size + j])
        label = np.copy(trainval_labels[i * grid_size + j])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        tile = cv2.rectangle(tile, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = tile

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %% [markdown]
# ### A bit about train-test
#
# You just downloaded a training and a test set.
#
# - We use the training set for forward/backward
# - We use the validation set to tune hyperparameters (optimizers, early stopping)
# - We use the test set for final metrics on our tuned model
#
# ![](https://i.stack.imgur.com/osBuF.png)
#
# For more information as to why we use train/validation and test refer to these articles:
#
# - https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
# - https://www.freecodecamp.org/news/what-to-do-when-your-training-and-testing-data-come-from-different-distributions-d89674c6ecd8/
# - https://kevinzakka.github.io/2016/09/26/applying-deep-learning/
#
# Here we will split the trainval_dataset to obtain a training and a validation dataset.
#
# For example, try to use 20% of the images as validation
#
# You must have seen that the dataset was really unbalanced, so a random sampling will not work...
#
# Use stratified sampling to keep the label distribution between training and validation
#
# We are also going to reduce the number of background examples to speedup trainings

# %%
background_indexes = np.where(trainval_labels == 0)[0][::3]
print(len(background_indexes))
foreground_indexes = np.where(trainval_labels == 1)[0]

train_bg_indexes = background_indexes[: int(0.8 * len(background_indexes))]
valid_bg_indexes = background_indexes[int(0.8 * len(background_indexes)) :]

train_fg_indexes = foreground_indexes[: int(0.8 * len(foreground_indexes))]
valid_fg_indexes = foreground_indexes[int(0.8 * len(foreground_indexes)) :]

train_indexes = list(train_bg_indexes) + list(train_fg_indexes)
valid_indexes = list(valid_bg_indexes) + list(valid_fg_indexes)

train_images = trainval_images[train_indexes, :, :, :]
train_labels = trainval_labels[train_indexes]

valid_images = trainval_images[valid_indexes, :, :, :]
valid_labels = trainval_labels[valid_indexes]

# %%
train_images.shape

# %%
print(np.unique(train_labels, return_counts=True))

# %%
print(np.unique(valid_labels, return_counts=True))

# %% [markdown]
# What is the mean of our data ? 
# Whats is the standard deviation ?

# %%
# Compute the dataset statistics in [0.,1.], we're going to use it to normalize our data

mean = np.mean(train_images, axis=(0, 1, 2)) / 255.0
std = np.std(train_images, axis=(0, 1, 2)) / 255.0

print("mean = ", mean)
print("std = ", std)

# %% [markdown]
# ## Preparing our training
#
# Remember that training a deep learning model requires:
#
# - Defining a model to train
# - Defining a loss function (cost function / criterion) to compute gradients with
# - Defining an optimizer to update parameters
# - Putting the model on the accelerator device that trains very fast (GPU, TPU)... You'll learn about GPUs later :)
#
# ![](https://pbs.twimg.com/media/E_1d06cVIAcYheX?format=jpg)
#
# The training loop is "quite basic" : We loop over samples of the dataset (in batches) several times over :
#
# ![](https://pbs.twimg.com/media/E_1d06XVcA8Dhzs?format=jpg)
#

# %%
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# %% [markdown]
# ### Defining Dataset & Transforms
#
# First, we need to tell pytorch how to load our data.
#
# Have a look at : https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#
# We write our own `torch.data.Dataset` class

# %%
class NpArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_transforms: Callable = None,
        label_transforms: Callable = None,
    ):
        self.images = images
        self.labels = labels
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index: int):
        x = self.images[index]
        y = self.labels[index]

        if self.image_transforms is not None:
            x = self.image_transforms(x)
        else:
            x = torch.tensor(x)

        if self.label_transforms is not None:
            y = self.label_transforms(y)
        else:
            y = torch.tensor(y)

        return x, y


# %% [markdown]
# Then we need to process our data (images) into "tensors" that torch can process, we define "transforms"

# %%
# transform to convert np array in range [0,255] to torch.Tensor [0.,1.]
# then normalize by doing x = (x - mean) / std
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# here we don't have anything to do
target_transforms = None

# %% [markdown]
# Now we put everything together into something to load our data

# %%
# load the training data
train_set = NpArrayDataset(
    images=train_images,
    labels=train_labels,
    image_transforms=image_transforms,
    label_transforms=target_transforms,
)

print(len(train_set))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# load the validation data
validation_set = NpArrayDataset(
    images=valid_images,
    labels=valid_labels,
    image_transforms=image_transforms,
    label_transforms=target_transforms,
)

print(len(validation_set))

val_loader = DataLoader(validation_set, batch_size=64, shuffle=True)

# %% [markdown]
# ### Check that your dataset outputs correct data
#
# Always do this as a sanity check to catch bugs in your data processing pipeline
#
# Write the inverse transformation by hand to ensure it's ok
#
# ![andrej](https://storage.googleapis.com/fchouteau-isae-deep-learning/static/andrej_tweet_1.png)

# %%
k = np.random.randint(len(train_set))
x, y = train_set[k]

# From torch
# Inverse transform
x = x.numpy()
x = x.transpose((1, 2, 0))
x = x * std + mean
x = x.clip(0.0, 1.0)
x = (x * 255.0).astype(np.uint8)

print("Inverse transform is OK ?")
print("Label {}".format(y))
plt.imshow(x)
plt.show()

plt.imshow(train_set.images[k])
plt.show()

# %% [markdown]
# ## Model

# %% [markdown]
# ### On which device will we train ?
#
# We will check if we have a GPU and set the "device" of pytorch on it so that it trains on GPU 

# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(DEVICE)

# %% [markdown]
# ### Defining a model and computing the parameters
#
# Now we have to define a CNN to train. It's usually called a "network", and we define its "architecture".
#
# Defining a good architecture is a huge field of research (a pandora's box) that takes a lot of time, but we can define "sane architectures" easily:
#
# Basically, CNN architectures are a stacks of :
# - Convolution layers + non linearities
# - Pooling layer
# - A final "activation" layer at the end (for classification) that allows us to output probabilities
#
# ![](https://cs231n.github.io/assets/cnn/convnet.jpeg)
#
# Let's define a model together:
#
# ```python
# model = nn.Sequential(
#     # A block of 2 convolutions + non linearities & a pooling layers
#     # IN SHAPE (3,64,64)
#     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
#     # OUT SHAPE (16,62,62)
#     nn.ReLU(),
#     # IN SHAPE (16,62,62)
#     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
#     # OUT SHAPE (16,60,60)
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     # OUT SHAPE (16,30,30)
#     # Another stack of these
#     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
#     # OUT SHAPE (?,?,?)
#     nn.ReLU(),
#     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#     nn.ReLU(),
#     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     # OUT SHAPE (?,?,?)
#     # Another stack of these
#     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#     # OUT SHAPE (?,?,?)
#     nn.ReLU(),
#     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#     # OUT SHAPE (?,?,?)
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     # OUT SHAPE (?,?,?)
#     # A final classifier
#     nn.Flatten(),
#     nn.Linear(in_features=4 * 4 * 64, out_features=256), # do you understand why 4 * 4 * 64 ?
#     nn.ReLU(),
#     nn.Dropout(p=0.25),
#     nn.Linear(in_features=256, out_features=64),
#     nn.ReLU(),
#     nn.Dropout(p=0.25),
#     nn.Linear(in_features=64, out_features=1),
#     nn.Sigmoid(),
# )
# ```
#
# **Questions**
#
# Knowing that the input image size is (3,64,64), go through the model step by step,
#
# Can you fill the blanks for the shapes ?
#
# Do you understand why ? 

# %%
# Let's test this !

some_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Another stack of these
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Another stack of these
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # A final classifier
    nn.Flatten(),
    nn.Linear(in_features=4 * 4 * 64, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(in_features=256, out_features=64),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(in_features=64, out_features=1),
    nn.Sigmoid(),
)

# We define an input of dimensions batch_size, channels, height, width
x = torch.rand((16, 3, 64, 64))

print(x.shape)

y = some_model(x)

print(y.shape)

# Let's visualize each shape using our summarize helper
print(summarize(some_model, example_input_array=x))

# let's delete the model now, we won't need it
del some_model

# %% [markdown]
# **Let's do it yourself !**
#
# About weight init :
# - https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
# - https://www.pyimagesearch.com/2021/05/06/understanding-weight-initialization-for-neural-networks/

# %%
# Let's define another model, except this time there are blanks ... it's up to you to fill them


def init_weights(model):
    for m in model.modules():
        # Initialize all convs
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")


def model_fn():
    model = nn.Sequential(
        # A first convolution block
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=..., out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Another stack of these
        nn.Conv2d(in_channels=..., out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=..., out_channels=..., kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # A final classifier
        nn.Flatten(),
        nn.Linear(in_features=... * ... * ..., out_features=64),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(in_features=64, out_features=1),
        nn.Sigmoid(),
    )

    return model


# %%
model = model_fn()

print(model)

x = torch.rand(
    (16, 3, 64, 64)
)  # We define an input of dimensions batch_size, channels, height, width

print(x.shape)

y = model(x)

print(y.shape)

print(summarize(model, example_input_array=x))

# THIS CELL SHOULD NOT GIVE AN ERROR !

# %% [markdown]
# Hint: The answer (and there can only be one) is :
#
# <details>
#     <summary>Solution</summary> 
#     
# ```python
#
# def model_fn():
#     model = nn.Sequential(
#         # A first convolution block
#         nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         # Another stack of these
#         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         # A final classifier
#         nn.Flatten(),
#         nn.Linear(in_features=12 * 12 * 32, out_features=64),
#         nn.ReLU(),
#         nn.Dropout(p=0.1),
#         nn.Linear(in_features=64, out_features=1),
#         nn.Sigmoid(),
#     )
#
#     _init_weights(model)
#
#     return model
#
#
# model = model_fn()
#
# print(model)
#
# x = torch.rand((16, 3, 64, 64))  # We define an input of dimensions batch_size, channels, height, width
#
# print(x.shape)
#
# y = model(x)
#
# print(y.shape)
#
# print(summarize(model,example_input_array=x))
# ```
#
# And outputs this
#
# ```
# Sequential(
#   (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
#   (3): ReLU()
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
#   (6): ReLU()
#   (7): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
#   (8): ReLU()
#   (9): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
#   (10): ReLU()
#   (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (12): Flatten(start_dim=1, end_dim=-1)
#   (13): Linear(in_features=4608, out_features=64, bias=True)
#   (14): ReLU()
#   (15): Dropout(p=0.1, inplace=False)
#   (16): Linear(in_features=64, out_features=1, bias=True)
#   (17): Sigmoid()
# )
# torch.Size([16, 3, 64, 64])
# torch.Size([16, 1])
#    | Name | Type      | Params | In sizes         | Out sizes       
# --------------------------------------------------------------------------
# 0  | 0    | Conv2d    | 448    | [16, 3, 64, 64]  | [16, 16, 62, 62]
# 1  | 1    | ReLU      | 0      | [16, 16, 62, 62] | [16, 16, 62, 62]
# 2  | 2    | Conv2d    | 2.3 K  | [16, 16, 62, 62] | [16, 16, 60, 60]
# 3  | 3    | ReLU      | 0      | [16, 16, 60, 60] | [16, 16, 60, 60]
# 4  | 4    | MaxPool2d | 0      | [16, 16, 60, 60] | [16, 16, 30, 30]
# 5  | 5    | Conv2d    | 4.6 K  | [16, 16, 30, 30] | [16, 32, 28, 28]
# 6  | 6    | ReLU      | 0      | [16, 32, 28, 28] | [16, 32, 28, 28]
# 7  | 7    | Conv2d    | 9.2 K  | [16, 32, 28, 28] | [16, 32, 26, 26]
# 8  | 8    | ReLU      | 0      | [16, 32, 26, 26] | [16, 32, 26, 26]
# 9  | 9    | Conv2d    | 9.2 K  | [16, 32, 26, 26] | [16, 32, 24, 24]
# 10 | 10   | ReLU      | 0      | [16, 32, 24, 24] | [16, 32, 24, 24]
# 11 | 11   | MaxPool2d | 0      | [16, 32, 24, 24] | [16, 32, 12, 12]
# 12 | 12   | Flatten   | 0      | [16, 32, 12, 12] | [16, 4608]      
# 13 | 13   | Linear    | 294 K  | [16, 4608]       | [16, 64]        
# 14 | 14   | ReLU      | 0      | [16, 64]         | [16, 64]        
# 15 | 15   | Dropout   | 0      | [16, 64]         | [16, 64]        
# 16 | 16   | Linear    | 65     | [16, 64]         | [16, 1]         
# 17 | 17   | Sigmoid   | 0      | [16, 1]          | [16, 1]         
# --------------------------------------------------------------------------
# 320 K     Trainable params
# 0         Non-trainable params
# 320 K     Total params
# 1.284     Total estimated model params size (MB)
# ```
#
# </details>
#
#
# You should be able to understand this

# %% {"id": "9gpZy_3cC1yk", "tags": ["solution"]}
def model_fn():
    model = nn.Sequential(
        # A first convolution block
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Another stack of these
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # A final classifier
        nn.Flatten(),
        nn.Linear(in_features=12 * 12 * 32, out_features=64),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(in_features=64, out_features=1),
        nn.Sigmoid(),
    )

    _init_weights(model)

    return model


model = model_fn()

print(model)

x = torch.rand(
    (16, 3, 64, 64)
)  # We define an input of dimensions batch_size, channels, height, width

print(x.shape)

y = model(x)

print(y.shape)

print(summarize(model, example_input_array=x))

# %% {"colab": {"base_uri": "https://localhost:8080/"}, "id": "DAv9FrjAC1yl", "outputId": "6b42440f-8806-41e3-dc97-ecb499de36bd"}
# moving model to gpu if available
model = model.to(DEVICE)

# %% [markdown] {"id": "LhpN-UNfC1yl"}
# ### Defining our loss and optimizer
#
# Check the definition of the binary cross entropy:
#
# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss

# %% {"id": "6w1BHLnoC1ym"}
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


# %% [markdown] {"id": "8d1qaMZ8C1ym"}
# ## Training with pytorch
#
# We will actually train the model, and plot training & validation metrics during training
#
# Be careful, if you train several times the same model it will continue optimizing its parameters
#
# Its advised to define a new model if you are executing the training loop several times

# %% [markdown]
# ### Defining the Training loop

# %%
def train_one_epoch(model, train_loader, opt, loss_fn):

    epoch_loss = []

    for i, batch in enumerate(train_loader):

        # get one batch
        x, y_true = batch
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)

        # format the y_true so that it is compatible with the loss
        y_true = y_true.view((-1, 1)).float()

        # zero the parameter gradients
        opt.zero_grad()

        # forward
        y_pred = model(x)

        # compute loss
        loss = loss_fn(y_pred, y_true)

        # backward
        loss.backward()

        # update parameters
        opt.step()

        # save statistics
        epoch_loss.append(loss.item())

        if i % 10 == 0:
            print(f"Batch {i}, curr loss = {loss.item():.03f}")

    return np.asarray(epoch_loss).mean()


def valid_one_epoch(model, valid_loader, loss_fn):

    epoch_loss = []

    for i, batch in enumerate(valid_loader):
        with torch.no_grad():
            # get one batch
            x, y_true = batch
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            # format the y_true so that it is compatible with the loss
            y_true = y_true.view((-1, 1)).float()

            # forward
            y_pred = model(x)

            # compute loss
            loss = loss_fn(y_pred, y_true)

            # save statistics
            epoch_loss.append(loss.item())

    return np.asarray(epoch_loss).mean()


# %% [markdown]
# ### Putting everything together to run a training
#
# Here we copy paste previous code so that you are sure you have setup everything correctly

# %%
model = model_fn()

# moving model to gpu if available
model = model.to(DEVICE)

print(model)

# We define an input of dimensions batch_size, channels, height, width
x = torch.rand((16, 3, 64, 64))

x = x.to(DEVICE)

print(x.shape)

y = model(x)

print(y.shape)

print(summarize(model, example_input_array=x))

# %% [markdown]
# **Hyperparameters**
#
# Here we define what we call hyperparameters, the "meta-parameters" of the training that you can modify to affect your training

# %%
EPOCHS = 10  # Set number of epochs, example 100
LEARNING_RATE = 1e-2
MOMENTUM = 0.9

# %%
# Define criterion and optimizer
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=1e-4
)

# %% [markdown]
# Let's train our network !

# %%
# Reinitialize our model
init_weights(model)
# Send model to GPU
model = model.to(DEVICE)

train_losses = []
valid_losses = []

# loop over the dataset multiple times
for epoch in range(EPOCHS):
    model.train()  # Set model in training mode
    train_epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    model.eval()  # Set model in eval mode
    valid_epoch_loss = valid_one_epoch(model, val_loader, criterion)

    print(f"EPOCH={epoch}, TRAIN={train_epoch_loss}, VAL={valid_epoch_loss}")

    train_losses.append(train_epoch_loss)
    valid_losses.append(valid_epoch_loss)

# %%
# Plot training / validation loss
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()

# %% [markdown] {"id": "e08d2219"}
# ### Training analysis
#
# How would you analyze your training ?
#
# Is it underfitting ?
#
# Is it overfitting ?

# %% [markdown] {"id": "3edb59c5"}
# ### Model saving
#
# There are several ways to save your model :
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# - torch.save: Saves a serialized object to disk. This function uses Python’s pickle utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.
#
# - torch.load: Uses pickle’s unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into (see Saving & Loading Model Across Devices).
#
# - torch.nn.Module.load_state_dict: Loads a model’s parameter dictionary using a deserialized state_dict. For more information on state_dict, see What is a state_dict?.
#
# - scripting / tracing the model: https://pytorch.org/docs/stable/jit.html
#
# The first 2 options require you to import the model definition as it uses pickle
# The third option requires you to redefine an empty model with the same architecture and load the weights, because we are only saving the "state" (e.g. parameters, weights, biases)
# The fourth option allow to make a "self-contained" model that can be used later, but comes with caveats

# %% [markdown] {"id": "afa82d48"}
# ### State dict saving 
#
# This is the recommended method because it allows to reuse the model with any code

# %% {"id": "cc4efcda"}
# State dict saving
with open("model.pt", "wb") as f:
    torch.save(model.state_dict(), f)

# Your model is saved here "/content/model.pt"
# See below for how to reload the model

# %% [markdown]
# To reload such a model, you have to instantiate an empty model with the same architecture then load the state dict (the weights)
# ```python
# # Instantiate a new empty model
# model = model_fn()
#
# print(model)
#
# # Load state
# checkpoint_path = "model.pt"
# model.load_state_dict(torch.load(checkpoint_path))
#
# print("Model Loaded")
# ```
#
# This is very nice because you get a model that you can finetune, retrain, modify. However, this means that you have to "port" the model definition code to production. 

# %% [markdown]
# ### Model scripting
#
# But for production, in order to avoid shipping the model definition code, we like to have an "self-contained" binary that we can deliver to the production team.
#
# Here we try to "script" the model, meaning that we compile the graph to a static version of itself.
#
# https://pytorch.org/docs/stable/jit.html

# %%
import torch.jit

# Put the model in eval mode
model = model.cpu().eval()

# Script the model
scripted_model = torch.jit.script(model)

# Save
scripted_model.save("scripted_model.pt")

# Your model is saved here "/content/scripted_model.pt"

print(scripted_model)

# Scripted model reloading (demo)
scripted_loaded_model = torch.jit.load("scripted_model.pt", map_location=DEVICE)

print(scripted_loaded_model)

# %% [markdown]
# ### Download the scripted model
#
# We are going to download the scripted model to be able to re-use it elsewhere (in another notebook for example), without having to rewrite the model definition function
#
# Uncomment this on google colab to download the model

# %%
# from google.colab import files

# files.download('scripted_model.pt')

# %% [markdown]
# We have finished what we need to do with the model, let's delete it !

# %%
del model

# %% [markdown]
# ## Testing our models and computing metrics
#
# Now that we have a trained network, it is important to measure how well it performs. We do not do that during training because theoretically we try to test on a context closer to how the final model will be used, meaning this can be another pipeline and is usually outside the training engine.
#
# You can refer to your ML course or on resources on the web to see how we can measure it.

# %% [markdown]
# ### Loading saved model
#
# State dict method

# %%
# Instantiate a new empty model
model = model_fn()

print(model)

# Load state
checkpoint_path = "model.pt"
model.load_state_dict(torch.load(checkpoint_path))

print("Model Loaded")

# %% [markdown]
# ### Inferencing on the test dataset
#
# Now we will run predictions on the test dataset using the newly loaded model

# %%
test_ds = NpArrayDataset(
    images=test_images,
    labels=test_labels,
    image_transforms=image_transforms,
    label_transforms=target_transforms,
)

# %%
import tqdm

# %%
y_true = []
y_pred = []

# Send model to correct device
model.to(DEVICE)

# Put model in evaluatio mode (very important)
model.eval()

# Disable all gradients things
with torch.no_grad():
    for x, y_t in tqdm.tqdm(test_ds, "predicting"):
        x = x.reshape((-1,) + x.shape)
        x = x.to(DEVICE)
        y = model.forward(x)
        y = y.to("cpu").numpy()

        y_t = int(y_t.to("cpu").numpy())

        y_pred.append(y)
        y_true.append(y_t)
y_pred = np.concatenate(y_pred, axis=0)
y_true = np.asarray(y_true)

# %%
# print(y_pred.shape)

print(y_pred[4])

# %%
y_pred_classes = y_pred[:, 0] > 0.5

# %% [markdown]
# ### Confusion matrix
# Here, we are first computing the [confusion matrix]():

# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

print("Confusion matrix")
cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["background", "aircraft"]
)

disp.plot()
plt.show()

# %% [markdown]
# ### ROC curve
#
# The next metric we are computing is the [Receiver Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html). A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The method was originally developed for operators of military radar receivers starting in 1941, which led to its name. 
#
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Roc_curve.svg/512px-Roc_curve.svg.png)
#
# ![](http://algolytics.com/wp-content/uploads/2018/05/roc1_en.png)
#
# It is used to choose a threshold on the output probability in case you are interesting in controling the false positive rate.
#
# In our case with our aircraft classifier :
#
#
# *   **True Positive Rate** = Number of well classified aircraft / Total number of aircraft
# *   **False Positive Rate** = Number of background misclassified as aircraft / Total number of background
#
#

# %%
# Compute ROC curve and Area Under Curve

from sklearn.metrics import auc, roc_curve

# We round predictions for better readability
y_pred_probas = np.round(y_pred[:, 0], 2)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probas)
roc_auc = auc(fpr, tpr)

# %%
plt.figure()
lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Using the ROC curve to select an optimal threshold
#
# The ROC curve can be used to select the best decision threshold for classifying an aircraft as positive.
#
# Plot the ROC curve with thresholds assigned to points in the curve (you can round the predictions for a simpler curve)

# %% {"tags": []}
# We round predictions every 0.05 for readability
y_pred_probas = (y_pred[:, 0] / 0.05).astype(np.int) * 0.05

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probas)
roc_auc = auc(fpr, tpr)

plt.clf()
fig = plt.figure(figsize=(10, 10))
plt.step(fpr, tpr, "bo", alpha=0.2, where="post")
plt.fill_between(fpr, tpr, alpha=0.2, color="b", step="post")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title("2-class ROC curve: AUC={:0.2f}".format(roc_auc))
plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
plt.grid()

for tp, fp, t in zip(tpr, fpr, thresholds):
    plt.annotate(
        np.round(t, 2),
        xy=(fp, tp),
        xytext=(fp + 0.05, tp - 0.05),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
    )
plt.savefig("roc_curve_thresholds.png")
plt.show()

# %%
max(y_pred_probas)

# %% [markdown]
# Now, choose a threshold on the curve where you miss less than 10% of the aircrafts

# %%
selected_threshold = ...

print("Confusion matrix")

y_pred_classes = y_pred_probas > selected_threshold

cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["background", "aircraft"]
)

disp.plot()
plt.show()

# How did the confusion matrix evolve ? Does it match your intuition ?

# %% [markdown]
# ### Misclassified examples
#
# It is always interesting to check mis classified examples.
#
# It usually provides tips on how to improve your model.

# %%
misclassified_idxs = np.where(y_pred_classes != y_true)[0]

print(len(misclassified_idxs))

print(misclassified_idxs)

misclassified_images = test_images[misclassified_idxs]
misclassified_true_labels = test_labels[misclassified_idxs]
misclassified_pred_labels = y_pred_classes[misclassified_idxs]

grid_size = 4
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        img = np.copy(misclassified_images[i * grid_size + j])
        pred = np.copy(misclassified_pred_labels[i * grid_size + j])
        color = (255, 127, 0) if pred == 1 else (255, 0, 0)
        tile = cv2.rectangle(img, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = img

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %% [markdown]
# ## Improving our training / validation loop
#
# We will add more advanced features to our training loop for better models
#
# Copy the train / valid loop and update it accordingly

# %% [markdown]
# ### Computing accuracy during training / validation
#
# Update the `valid_one_epoch` to compute accuracy during the validation loop, and plot its evolution during training
#
# Use the ROC curve computation where we compute the pred / true classes as inspiration
#
# Here's an example (that needs to be modified)
# ```python
#
# correct_pred = 0
# total_pred = 0
# with torch.no_grad():
#     for data in valid_loader:
#         images, labels = data
#         outputs = net(images)
#         predictions = torch.round(outputs)[:,0]
#         # collect the correct predictions
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred += 1
#             total_pred += 1
#             
#     # print accuracy
#     accuracy = 100 * (correct_pred / total_pred)
#     print("Accuracy is: {:.1f} %".format(accuracy))
#
# ```                                             

# %% [markdown]
# ### Early stopping
#
# You may have seen that it is possible to overfit it you're not careful.
# Thanks to train_epoch_loss and valid_epoch_loss, it is possible to prevent overtiftting by stopping network training when those 2 losses diverge.
#
# **Go back to your previous class and adapt the training loop to add early stopping**

# %% [markdown] {"id": "5f2ef879"}
# ### Data Augmentation
#
#
# One technique for training CNNs on images is to put your training data through data augmentation to generate similar-but-different examples to make your network more robust.
#
# You can generate "augmented images" on the fly or use composition to generate data
#
# - We are going to wrap our numpy arrays with `torch.utils.data.Dataset` class
#
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
#
# - Here is how we use torch Compose to augment data
#
# https://pytorch.org/docs/stable/torchvision/transforms.html
#
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms
#
# Note: This step requires a bit of tinkering from numpy arrays to torch datasets, it's fine if you skip it. For the next notebook it may prove a useful way of gaining performance
#
# **Remember : We apply data augmentation only during training**
#

# %%
import torch.functional as F
import torch.utils
import torchvision.transforms

# %%
# Example (very simple) data augmentation to get your started, you can add more transforms to this list

train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
)

# %%
trainset_augmented = NpArrayDataset(
    images=train_images,
    labels=train_labels,
    image_transforms=train_transform,
    label_transforms=None,
)

# %%
# Get image from dataset. Note: it has been converted as a torch tensor in CHW format in float32 normalized !
img, label = trainset_augmented[0]
img = img.numpy().transpose((1, 2, 0)) * std + mean
img = img.clip(0.0, 1.0)
img = (img * 255.0).astype(np.uint8)
plt.imshow(img)
plt.show()

# Compare effects of data augmentation (Random flips)
img_orig = trainset_augmented.images[0]
plt.imshow(img_orig)
plt.show()

# %%
# do another training and plot our metrics again. Did we change something ?

# %% [markdown] {"tags": []}
# ### Best checkpoint
#
# You've seen how to save model checkpoint. However we saved the model at the end of training. What if there is an issue (like overfitting ? or our computer crashes !!!) ? 
#
# How to keep a good copy of our model at any point ? 
#
# The idea is that during the training, we always save the checkpoint with the lowest valid loss, then reload it at the end of training
#
# **Modify the train loop to keep the best model state dict at any point, then reload it at the end of training**
#

# %%

# %% [markdown]
# ### Food for thoughts: Tooling
#
# To conclude this notebook, reflect on the following,
#
# You have launched different experiences and obtained different results,
#
# Did you feel the notebook you used was sufficient ? Which tools would you like to have in order to properly run your experiments ? (Quick google search or ask someone) Do they already exist ?
#
# ### **Presentation : High level frameworks**
#
# <img src="https://raw.githubusercontent.com/pytorch/ignite/master/assets/logo/ignite_logo_mixed.svg" alt="ignite" style="width: 400px;"/>
#
# Pytorch ignite is what we call a "high-level library" over pytorch, its objectives is to abstract away most of the boilerplate code for training deep neural network.
#
# Usually, they make the development process easier by enabling you to focus on what's important instead of writing distributed and optimized training loops and plugging metrics / callbacks. Because we all forgot to call `.backward()` or `.zero_grad()` at least once.
#
# Here an overview of the high-level libraries available for pytorch,
#
# https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem?utm_source=twitter&utm_medium=tweet&utm_campaign=blog-model-training-libraries-pytorch-ecosystem
#
# Of these, we would like to highlight three of them:
#
# - pytorch-ignite, officially sanctioned by the pytorch team (its repo lives at https://pytorch.org/ignite/), which is developped by [someone from Toulouse](https://twitter.com/vfdev_5) - yes there is a member of the pytorch team living in Toulouse, we are not THAT behind in ML/DL :wishful-thinking:
#
# - pytorch-lightning (https://www.pytorchlightning.ai/) which has recently seen its 1.0 milestone and has been developped to a company. It is more "research oriented" that pytorch-ignite, and with a lower abstraction level, but seems to enable more use case.
#
# - catalyst (https://github.com/catalyst-team/catalyst) 
#
# - skorch (https://github.com/skorch-dev/skorch). This class was previously written in skorch. Skorch mimics the scikit-learn API and allows bridging the two libraries together. It's a bit less powerful but you write much less code than the two libraries above, and if you are very familiar with scikit-learn, it could be very useful for fast prototyping
#
#
# **Take a look at the [previous class](https://nbviewer.jupyter.org/github/SupaeroDataScience/deep-learning/blob/main/deep/PyTorch%20Ignite.ipynb), the [official examples](https://nbviewer.jupyter.org/github/pytorch/ignite/tree/master/examples/notebooks/) or the [documentation](https://pytorch.org/ignite/) if want to learn about Ignite**

# %% [markdown]
# ## **Optional** exercises to run at home
#
# If you're done with this, you can explore a little bit more : Now that we have a nice training loop we can do hyperparameter tuning !
#
# As you can see, there are a lot of parameters we can choose:
#
# - the optimizer
# - the learning rate
# - the model architecture
#  
# etc... !
#
#
# - Try to play with network hyperparameters. The dataset is small and allow fast iterations so use it to have an idea on hyperparameter sensitivity.
#     number of convolutions, other network structures, learning rates, optimizers,...
#
# - Example: Compare again SGD and ADAM
#
# - Try to use the ROC curve to select a threshold to filter only negative examples without losing any positive examples
#
# When you are done with the warmup, go to the next notebook. But remember that next datasets will be larger and you will not have the time (trainings will take longer ) to experiment on hyperparameters.
#
# **You can try more things**

# %% [markdown]
# ### Optimizer Changes
# Change the optimizer from SGD to optim.Adam. Is it better ? 

# %%
# HERE

# %% [markdown]
# ### Batch Normalization
#
# One of the most used "layer" beyond conv / pool / relu is "batch normalization",
#
# http://d2l.ai/chapter_convolutional-modern/batch-norm.html
#
# Try adding it to your network and see what happens !
#
# <details>
#
# ```python
# def model_fn():
#     model = nn.Sequential(
#         # A first convolution block
#         nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
#         nn.BatchNorm2d(16),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
#         nn.BatchNorm2d(16),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         # Another stack of these
#         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         # A final classifier
#         nn.Flatten(),
#         nn.Linear(in_features=12 * 12 * 32, out_features=64),
#         nn.BatchNorm1d(64),
#         nn.ReLU(),
#         nn.Dropout(p=0.1),
#         nn.Linear(in_features=64, out_features=1),
#         nn.Sigmoid(),
#     )
#
#     return model
# ```
#     
# </details>

# %% [markdown]
# ### Trying other models
#
# You have seen a class on different model structure,
# https://supaerodatascience.github.io/deep-learning/slides/2_architectures.html#/
#
# Now is the time to try and implement them. 
#
# For example, try to write a VGG-11 with fewer filters by yourself... or a very small resnet using [this](https://github.com/a-martyn/resnet/blob/master/resnet.py) as inspiration
#
# You can also use models from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html#classification) in your loop, or as inspiration
#

# %% [markdown]
# **Modify the model structure and launch another training... Is it better ?**

# %%
# HERE

# %% [markdown] {"tags": []}
# ### LR Scheduling
#
# Sometimes it's best to reduce the learning rate if you stop improving, or to reduce learning rate at the end of training
#
# Tutorial : https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/#top-basic-learning-rate-schedules
#
# - **Modify the train loop to change the learning rate when the validation loss is stagnating**
#
# - **Modify the train loop to change the learning rate when the validation loss is stagnating**

# %%
# ...

# %% [markdown]
# ### Transfer Learning
#
# For usual tasks such as classification or detection, we use "transfer learning":
#
#     In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.
#     
# Adapt this tutorial to do transfer learning from a network available in torchvision to our use case
#
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#
# I advise you to select resnet18
#
# The biggest library of pretrained models is available here :
#
# https://github.com/rwightman/pytorch-image-models

# %%
