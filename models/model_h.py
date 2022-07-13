import torch
import torch.nn as nn
import torchvision.models as models
# from utils import *
import torch.nn.functional as func

class HashingNet(nn.Module):
    def __init__(self, nclass, nbit):
        super(HashingNet, self).__init__()

        self.encoderIMG_1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
        )
        self.encoderIMG_2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Dropout(0.4), # add
        )
        self.encoderIMG_3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
        )
        self.hash = nn.Sequential(
            nn.Linear(1024, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh(),
        )
        self.decoderIMG = nn.Sequential(
            nn.Linear(nbit, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096)
        )

    def forward(self, x, proto_F):
        proto_f1 = proto_F['F1_prototype'].cuda().to(torch.float)
        # proto_f2 = proto_F['F2_prototype'].cuda().to(torch.float)
        # proto_f3 = proto_F['F3_prototype'].cuda().to(torch.float)
        # proto_h = proto_F['H_prototype'].cuda().to(torch.float)

        # (1)
        # x = self.attention(x, proto_f1)      # attention 1
        f1 = self.encoderIMG_1(x)

        # (2)
        f1 = self.attention(f1, proto_f1)     # attention 2
        f2 = self.encoderIMG_2(f1)

        # (3)
        # f2 = self.attention(f2, proto_f2)     # attention 3
        f3 = self.encoderIMG_3(f2)

        # (4)
        # f3 = self.attention(f3, proto_f3)     # attention 4
        hashcode = self.hash(f3)

        # (5)
        # hashcode = self.attention(hashcode, proto_h)  # attention h
        feat_reconst = self.decoderIMG(hashcode)

        return f1, f2, f3, hashcode, feat_reconst

    def attention(self, x, feature_proto):
        # try:
        #     if x_.requires_grad == True:
        #         x = x_.clone().detach()
        #     else:
        #         x = x_.clone()
        # except Exception:
        #     x = x_

        # attention
        # x_ = x / torch.sqrt(torch.sum(x ** 2))  # feat_ normalize
        # feature_proto_ = feature_proto / torch.sqrt(torch.sum(feature_proto ** 2))  # feature_proto_ normalize

        # x_ = func.normalize(x)
        # feature_proto_ = func.normalize(feature_proto)
        # feat_ = self.att_1(x)
        # feature_proto_ = self.att_2(feature_proto)

        lambd1 = feature_proto.size(dim=1) * 2
        lambd2 = 0.5
        mask_img = torch.tanh((x.mm(feature_proto.t()) / lambd1))
        # mask_img = x_.mm(feature_proto_.t())
        # mask_img = nn.Softmax()(mask_img)

        # mask_img_ = nn.ReLU()(mask_img - mask_img.mean()) + mask_img.mean()#
        mask_f_x = mask_img.mm(feature_proto) /mask_img.sum(dim=1).unsqueeze(-1)
        # feat_att = lambd2 * x + (1 - lambd2) * mask_f_x
        feat_att =  lambd2 * x +  (1 - lambd2) * mask_f_x

        return feat_att
