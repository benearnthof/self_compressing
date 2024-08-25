"""
Trainer Class that handles training, evaluation, & checkpointing.
Evaluation needs to incorporate both performance statistics & model size in bits.
"""

from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from models import Net
from modules import QConv2d

import functools
from tqdm import trange

device = "cuda"

mnist_transform = T.Compose([
    T.ToTensor()
])

ds = MNIST(
    root="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data",
    download=True,
    transform = mnist_transform
)

ds_test = MNIST(
    root="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/data",
    download=True,
    transform = mnist_transform,
    train=False
)

dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)
x_test, y_test = next(iter(dl_test))

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

dl = cycle(dl)

# yields list of two tensors with batch of images at 0 and batch of labels at 1


model = Net()
model.to(device)
optimizer = Adam(model.parameters(), lr=3e-4)
test_accs, bytes_used = [], []
weight_count = sum(t.numel() for t in model.parameters())
lossfunction = CrossEntropyLoss()
len(optimizer.state_dict()["param_groups"][0]["params"]), weight_count

def train_step():
    optimizer.zero_grad()
    samples, targets = next(dl)
    samples, targets = samples.to(device), targets.to(device)
    pred = model(samples)
    loss = lossfunction(pred, targets)
    Q = functools.reduce(lambda x,y: x+y, [l.qbits() for l in model.modules() if isinstance(l, QConv2d)]) / weight_count
    loss = loss + 0.05 * Q # hyperparameter determines compression vs acc
    loss.backward()
    optimizer.step()
    return loss, Q


def get_test_acc(): 
    # directly convert to percent
    return (model(x_test.to(device)).argmax(axis=1)== y_test.to(device)).int().float().mean() * 100

model.to(device)
model.train()

test_acc = float("nan")
for i in (t:=trange(20000)):
    loss, Q = train_step()
    model_bytes = Q.item()/8*weight_count
    if i%10 == 9:
        test_acc = get_test_acc().item()
    test_accs.append(test_acc)
    bytes_used.append(model_bytes)
    t.set_description(f"loss: {loss.item():6.2f}  bytes: {model_bytes:.1f}  acc: {test_acc:5.2f}%")

# save performance metrics to disk
import pickle
with open('test_accs.pickle', 'wb') as handle:
    pickle.dump(test_accs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bytes_used.pickle', 'wb') as handle:
    pickle.dump(bytes_used, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_accs.pickle', 'rb') as handle:
    test_accs = pickle.load(handle)

with open('bytes_used.pickle', 'rb') as handle:
    bytes_used = pickle.load(handle)

from matplotlib import pyplot as plt
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.set_ylabel("Model Size (bytes)")
ax1.yaxis.label.set_color("red")
ax1.plot(bytes_used, color="red")
ax2 = ax1.twinx()
ax2.plot(test_accs, color="blue")
plt.ylim(80, 100)
ax2.set_ylabel("Test Accuracy (%)")
ax2.yaxis.label.set_color("blue")

fig.savefig("mnist.png")