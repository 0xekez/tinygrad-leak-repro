# To train the neural cellular automata, we need a loss function which
# measures the extent two images have the same "vibe". This loss
# function moves the target and generated image through the first five
# convolutional blocks of a trained VGG16 image classifier. For each
# pair of target (X) and actual (Y) convolutional block values, the
# loss is computed as:
#
# 1. X = sqrt(X+1)-1, Not sure why this vs regular euclidian distance
# 2. X = X.T@X / X.width / X.height, to compute the Gram matrix.
# 3. apply (1) and (2) to Y
# 4. error = (Y-X)^2
# 5. error = mean_across_all_channels(error)
#
# The total error across all convolutional block outputs is the sum of
# errors.

# How to implement this
# 1. VGG16 which returns intermediate layer activations for input image
# 2. Loss calculation routine

from tinygrad.tensor import Tensor

class Conv3x3:
    def __init__(self, in_channels, out_channels):
        self.weight = Tensor.uniform(out_channels, in_channels, 3, 3)
        self.bias = Tensor.uniform(out_channels)
    def __call__(self, x):
        return x.conv2d(self.weight).add(self.bias.reshape(1, -1, 1, 1))

class MaxPool2D:
    def __init__(self):
        return
    def __call__(self, x):
        return x.max_pool2d(kernel_size=2, stride=2, dilation=1)

class Relu:
    def __call__(self, x):
        return x.relu()

class VGG16Loss:
    def __init__(self):
        # Of interest are the activations of the first five
        # convolutions, so that's all the model contains.
        self.features = [
            Conv3x3(3, 64),    # (1)
            Relu(),
            Conv3x3(64, 64),   # (2)
            Relu(),
            MaxPool2D(),
            Conv3x3(64, 128),  # (3)
            Relu(),
            Conv3x3(128, 128), # (4)
            Relu(),
            MaxPool2D(),
            Conv3x3(128, 256), # (5)
        ]
        self.load_from_torchvision()

    def load_from_torchvision(self):
        import torchvision
        import torch
        from tinygrad.nn.state import get_state_dict
        weights = torchvision.models.vgg16(weights='DEFAULT').state_dict()
        for k,v in get_state_dict(self).items():
            v.assign(weights[k].numpy()).realize()

    def __call__(self, x):
        r=[]
        for f in self.features:
            x = f(x)
            if isinstance(f, Conv3x3):
                r.append(x)
        return r

def gram(a):
    s = a.shape
    # (bs, channels, width, height) => (bs, channels, width*height)
    a = a.reshape(s[0], s[1], -1)
    # In Self-Organising Textures they say they use L_2 norm
    # (euclidian distance) as loss, but in actual implementation they
    # take the square root here. Of course, square root of negative
    # numbers is NaN, and there are negative numbers in the
    # activations. So for now I'm just doing regular euclidian
    # distance.

    # a = (a+1).sqrt()-1
    gram = Tensor.einsum('bid,bjd->bij', a, a)
    return gram / (s[-1]*s[-2])

# Computes loss. Input should have shape (bs, channels, width,
# height)
#
# 1. X = sqrt(X+1)-1, Not sure why this vs regular euclidian distance
# 2. X = X.T@X / X.width / X.height, to compute the Gram matrix.
# 3. apply (1) and (2) to Y
# 4. error = (Y-X)^2
# 5. error = mean_across_all_channels(error)
def loss(a, b):
    losses = [(gram(b)-gram(a)).square().sum().sqrt() for a, b in zip(a, b)]
    loss = Tensor.stack(losses).mean()
    return loss
