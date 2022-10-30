import numpy as np
import torch


class Dojo:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
    ):
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.device = device

    def train(self):
        train_losses = []
        val_losses = []

        self.model.train()
        for i, batch in enumerate(self.train_dl):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            out = self.model(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            if i % 100 == 0:
                print("Training loss at step {}: {}".format(i, loss.item()))

        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, batch in enumerate(self.val_dl):
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                out = self.model(x)
                loss = self.loss_fn(out, y)
                acc = torch.sum(torch.argmax(out, dim=-1) == y)
                correct += acc.item()
                total += len(batch[1])
                val_losses.append(loss.item())

        val_acc = correct / total

        return np.mean(train_losses), np.mean(val_losses), val_acc

    def test(self):
        num_correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                y_hat = self.model(x)
                acc = torch.sum(torch.argmax(y_hat, dim=-1) == y)
                num_correct += acc.item()
                total += len(batch[1])

        return float(num_correct) / float(total)
