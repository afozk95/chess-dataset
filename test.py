import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a1 = nn.Conv2d(19, 64, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.b1 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.b2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.b3 = nn.Conv2d(256, 256, kernel_size=4, padding=0)

        self.linear = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        x = x.view(-1, 256)
        x = self.linear(x)

        return x


model = Net()
model.load_state_dict(torch.load("data/nets/value_60.pth", map_location=torch.device("cpu")))
model.eval()
# model.cuda()


import chess
from serializer import ChessPositionSerializer
board = chess.Board("rnbqkb1r/1ppppp1p/p4np1/3N4/4Q3/4P3/PPPPBPPP/RNB1K2R w KQkq - 2 8")
s = ChessPositionSerializer()

w = torch.Tensor([-3, -2, -1, 0, 1, 2, 3])
for move in board.legal_moves:
    board.push(move)
    board_s = torch.Tensor(s.serialize(board)).unsqueeze(0)
    board_o = (F.softmax(model(board_s)) * 100)
    print(move)
    print(board_o)
    board.pop()
