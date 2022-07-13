import torch.nn as nn
import numpy as np
import torch
from models.model_h import *
from utils import *
import torch.nn.functional as func
import pickle as pkl
import tqdm

def TrainHashing(args, loader):
    # 1. define modal todo
    hashing_net = HashingNet(args.nclass, args.nbit)

    # 1. init modal
    load_path = 'checkpoints/' + args.dataset + '/' + str(args.nbit)
    hashing_net.encoderIMG_1.load_state_dict(torch.load(os.path.join(load_path, 'PN_1.pth')))
    hashing_net.encoderIMG_2.load_state_dict(torch.load(os.path.join(load_path, 'PN_2.pth')))
    hashing_net.encoderIMG_3.load_state_dict(torch.load(os.path.join(load_path, 'PN_3.pth')))
    hashing_net.hash.load_state_dict(torch.load(os.path.join(load_path, 'PN_h.pth')))
    hashing_net.cuda()

    # 2. define loss
    criterion_l2 = nn.MSELoss().cuda()

    # 3. define optimizer
    optimizer = torch.optim.Adam([
        # {'params': hashing_net.encoderIMG_1.parameters()},
        # {'params': hashing_net.encoderIMG_2.parameters()},
        {'params': hashing_net.encoderIMG_3.parameters()},
        {'params': hashing_net.hash.parameters()},
        {'params': hashing_net.decoderIMG.parameters()},
    ], args.lr)

    # optimizer = torch.optim.Adam(hashing_net.parameters(), args.lr)

    # 4. get prototype
    if args.dataset == 'coco':
        proto_F_path = './checkpoints/coco/proto_' + str(args.nbit) + 'bit.pth' # type: torch
    elif args.dataset == 'nus21':
        proto_F_path = './checkpoints/nus21/proto_' + str(args.nbit) + 'bit.pth' # type: torch

    # load feat proto code
    # include F1_prototype F2_prototype F3_prototype H_prototype
    proto_F = torch.load(proto_F_path)

    # 6. train model
    hashing_net.train()
    for epoch in range(args.num_epoch):
        # for step, (img, label, plabel, index) in tqdm.tqdm(enumerate(loader)):
        for step, (img, label, plabel, index) in enumerate(loader):
            img = img.cuda().to(torch.float)
            optimizer.zero_grad()

            f1, f2, f3, h, feat_reconst = hashing_net(img, proto_F)
            b = torch.sign(h)

            # (1) 关系重构loss
            F_I = func.normalize(img)
            S_I = F_I.mm(F_I.t()) # [0, 1]
            S_I = S_I * 2 - 1 # [-1, 1]

            S_I = 0.7 * S_I + 0.3 * S_I.mm(S_I) / S_I.size(0) # [-1, 1]
            S_I_batch = S_I * 1.2 # [-1.2, 1.2]

            h_norm = func.normalize(h)
            S_h = h_norm.mm(h_norm.t()) # [-1, 1]

            relation_recons_loss = criterion_l2(S_h, S_I_batch) * args.lamda1

            # (2) 量化loss
            sign_loss = criterion_l2(h, b) * args.lamda2

            #(3) 重构loss
            semantic_recons_loss = criterion_l2(feat_reconst, img) * args.lamda3

            # total loss
            loss = relation_recons_loss + sign_loss + semantic_recons_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0 and (step + 1) == len(loader):
                print('Epoch [%3d/%3d]: Total Loss: %.4f, loss1: %.4f, loss2: %.4f, loss3: %.4f' % (
                    epoch + 1, args.num_epoch,
                    loss.item(),
                    relation_recons_loss.item(),
                    sign_loss.item(),
                    semantic_recons_loss.item()
                ))

    return hashing_net



