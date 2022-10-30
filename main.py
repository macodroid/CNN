import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import netron

from CnnModelCifar10 import CnnModelCifar10

classes = ('plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# for reproducibility
trainset, valset = torch.utils.data.random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

batch_size = 64

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

for X, y in val_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    inp_shape = X.shape
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = CnnModelCifar10(input_shape=inp_shape)
model.to(device)
print(model)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# scheduler = StepLR(optimizer, step_size=1, gamma=0.8)


def train(model, loss_fn, optimizer, train_dataloader, val_dataloader, verbose=True):
    train_losses = []
    val_losses = []

    model.train()
    for i, batch in enumerate(train_dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        if i % 100 == 0 and verbose:
            print("Training loss at step {}: {}".format(i, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(val_dataloader):
            x, y = batch[0].to(device), batch[1].to(device)

            out = model(x)
            loss = loss_fn(out, y)
            acc = torch.sum(torch.argmax(out, dim=-1) == y)
            correct += acc.item()
            total += len(batch[1])
            val_losses.append(loss.item())

    val_acc = correct / total

    return np.mean(train_losses), np.mean(val_losses), val_acc


def test(model, test_dataloader, verbose=True):
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x, y = batch[0].to(device), batch[1].to(device)

            y_hat = model(x)
            acc = torch.sum(torch.argmax(y_hat, dim=-1) == y)
            num_correct += acc.item()
            total += len(batch[1])

    return float(num_correct) / float(total)


epochs = 5

epoch_train_losses = []
epoch_val_losses = []
epoch_val_accs = []

for e in range(epochs):
    print(f"Epoch {e + 1}\n-------------------------------")
    train_loss, val_loss, val_acc = train(model=model, loss_fn=loss_fn, train_dataloader=train_dataloader,
                                          val_dataloader=val_dataloader, optimizer=optimizer)
    print("Val loss at epoch {}: {}".format(e, val_loss))
    print("Val acc at epoch {}: {}".format(e, val_acc))
    epoch_train_losses.append(train_loss)
    epoch_val_losses.append(val_loss)
    epoch_val_accs.append(val_acc)
    # scheduler.step()
print(f"Test Accuracy of the model: {test(model=model, test_dataloader=test_dataloader) * 100:.2f}")
# TODO running some experiments need to change name of the snapshot of model
torch.save(model, 'baseline_cifar10.pt')
plt.plot(epoch_train_losses, c='r')
plt.plot(epoch_val_losses, c='b')
plt.legend(['Train_loss', 'Val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs. Validation loss")
# plt.show()
# TODO same here change naming of plot
plt.savefig("baseline.png")
print("Done!")
