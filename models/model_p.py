import torch
import torch.nn as nn
import torchvision.models as models

class PrototypeNet(nn.Module):
    def __init__(self, nclass, nbit):
        super(PrototypeNet, self).__init__()

        self.encoderIMG_1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.encoderIMG_2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # add
        )
        self.encoderIMG_3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.hash = nn.Sequential(
            nn.Linear(1024, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh(),
        )

        self.netclassifier = nn.Sequential(
            nn.Linear(nbit, nclass)
        )

    def forward(self, x):
        f1 = self.encoderIMG_1(x)
        f2 = self.encoderIMG_2(f1)
        f3 = self.encoderIMG_3(f2)
        hashcode = self.hash(f3)
        predict_y = self.netclassifier(hashcode)

        return f1, f2, f3, hashcode, predict_y