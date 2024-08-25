"""
Trainer Class that handles training, evaluation, & checkpointing.
Evaluation needs to incorporate both performance statistics & model size in bits.
"""

from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader

mnist_transform = T.Compose([
    T.ToTensor()
])

ds = MNIST(
    root="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data",
    download=True,
    transform = mnist_transform
)

dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

dl = cycle(dl)
# yields list of two tensors with batch of images at 0 and batch of labels at 1
