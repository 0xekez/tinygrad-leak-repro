import random

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam

from vggloss import VGG16Loss, loss
from batch import SamplePool

# No leak happens if run on CPU.

# from tinygrad.device import Device
# Device.DEFAULT = 'CPU'

sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
],dtype=np.float32) / 8
sobelY = sobelX.T
laplacian = np.array([
    [1,   2, 1],
    [2, -12, 2],
    [1,   2, 1],
],dtype=np.float32) / 8

CHANNELS=8

def stocastic(x, p=0.5):
    return x * (Tensor.rand(*x.shape) >= p)

class Convolution:
    def __init__(self, kernel, channels):
        w, h = kernel.shape
        self.weight = kernel = Tensor(kernel).reshape(1, 1, w, h).repeat((channels, channels, 1, 1))
    def __call__(self, x): # x: (bs, channels, width, height)

        # Want to do a 2d convolution with wraparound, because want
        # the texture to repeat. To accomplish this, pad each side of
        # the input with one cell from the opoosite side. Then, run
        # the convolution on this padded value (so the output shape is
        # unchanged, as a 3x3 kernel only looks one cell away).
        x = Tensor.cat(x[:,:,-1:],x,x[:,:,:1],dim=2)
        x = Tensor.cat(x[:,:,:,-1:],x,x[:,:,:,:1], dim=3)
        return x.conv2d(self.weight, padding=0)

class NCA:
    def __init__(self, channels):
        self.gradX = Convolution(sobelX, channels)
        self.gradY = Convolution(sobelY, channels)
        self.laplacian = Convolution(laplacian, channels)
        self.l1 = Linear(channels*4, channels*4*2, bias=True)
        self.l2 = Linear(channels*4*2, channels, bias=True)

        # Semi-obscure bit from paper where they start the network in
        # a "do nothing" state. Adding this detail fixed an issue
        # where the models weights would become NaN after after more
        # than 10 training steps.
        self.l2.weight = Tensor.zeros(*self.l2.weight.shape)
        self.l2.bias = Tensor.zeros(*self.l2.bias.shape)
    def __call__(self, x):
        dx = self.gradX(x)
        dy = self.gradY(x)
        l = self.laplacian(x)
        features = Tensor.cat(x, dx, dy, l, dim=1)
        # (1, 64, width, height) => (1, width, height, 64)
        features = features.permute(0, 2, 3, 1)
        return self.l2(self.l1(features).relu()).permute(0, 3, 1, 2)

def show_img(x):
    import matplotlib.pyplot as plt
    plt.imshow(x)
    plt.axis('off')
    plt.show()

def render(model, steps=7, channels=CHANNELS):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    x = Tensor.zeros(1, channels, 224, 224)
    for _ in range(steps):
        x = x + stocastic(model(x))
    x = x[0][:3,:,:] # take RGB channels
    x = x.permute(1, 2, 0) # (height, width, channels)
    x = x.numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    show_img(x)
    return x

def load_img(img):
    from PIL import Image
    img = Image.open(img)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    # normalize using the same params as the torchvision model.
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return Tensor([img.transpose()])

def train(
        texture, # (1, 3, width, height)
        steps=2000,
        bs=4,
        channels=CHANNELS,
        nca=NCA(CHANNELS),
):
    width = texture.shape[-2]
    height = texture.shape[-1]
    do_n = lambda f, x, n: x if n==0 else do_n(f, f(x), n-1)

    Tensor.training=True
    pool = SamplePool(1024, channels, width, height)
    vgg16 = VGG16Loss()
    opt = Adam([nca.l1.weight, nca.l1.bias, nca.l2.weight, nca.l2.bias])
    y = vgg16(texture)
    for i in range(steps+1):
        batch = pool.sample(bs)
        batch = do_n(lambda x: x + stocastic(nca(x)), batch, random.choice(range(5,7)))
        x = vgg16(batch[:, :3, :, :])
        error = loss(x, y)
        if i%10==0:
            print(i, error.numpy())
            save_model(nca)
        opt.zero_grad()
        error.backward()
        opt.step()
        pool.commit(batch)
    Tensor.training=False
    return nca

from tinygrad.nn.state import safe_save, get_state_dict
def save_model(m):
    safe_save(get_state_dict(m), "model.safetensors")

im=load_img("marbled_0100.jpg")
model = train(im)
