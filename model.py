import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim


class MyDataset(Dataset):
    def __init__(self):
        dat = np.load("data/other/dataset.npz")
        self.X = dat["X"] 
        self.y = torch.tensor([self._parse_y(v)+3 for v in dat["y"].tolist()])

    def _parse_y(self, val):
        if val == 100:
            return 3
        elif val == -100:
            return -3
        elif val > 3.0:
            return 2
        elif val > 1.0:
            return 1
        elif val > -1.0:
            return 0
        elif val > -3.0:
            return -1
        else:
            return -2

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a1 = nn.Conv2d(19, 64, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.b1 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.b2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.b3 = nn.Conv2d(128, 128, kernel_size=4, padding=0)

        self.linear = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        x = x.view(-1, 128)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    device = "cuda"

    chess_dataset = MyDataset()
    train_ratio = 0.8
    train_length = int(len(chess_dataset) * train_ratio)
    test_length = len(chess_dataset) - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(chess_dataset, (train_length, test_length))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=True)
    model = Net()
    optimizer = optim.Adam(model.parameters())
    floss = nn.CrossEntropyLoss(reduction="sum")

    if device == "cuda":
        model.cuda()

    model.train()

    for epoch in range(1, 101):
        epoch_loss = 0
        epoch_correct = 0
        num_examples_epoch = 0
        print(f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            print(f"Batch {batch_idx}")
            data, target = data.to(device), target.to(device)
            data = data.float()

            optimizer.zero_grad()
            output = model(data)

            loss = floss(output, target)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            batch_correct = (output.max(1).indices == target).sum().item()
            num_examples_batch = target.shape[0]
            epoch_loss += batch_loss
            epoch_correct += batch_correct
            num_examples_epoch += num_examples_batch
            
            print(f"loss in epoch: {epoch_loss/num_examples_epoch}")
            print(f"accuracy in epoch: {epoch_correct/num_examples_epoch}")
            print(f"loss in batch: {batch_loss/num_examples_batch}")
            print(f"accuracy in batch: {batch_correct/num_examples_batch}")

        torch.save(model.state_dict(), f"data/nets/value_{epoch}.pth")
