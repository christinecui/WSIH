import torch.nn as nn
import numpy as np
import torch
import tqdm

from models.model_p import *
from utils import *
import torch.nn.functional as func
from load_data import *
from sklearn.cluster import KMeans

def getPrototype(args, loader):
    # 1. define modal
    prototype_net = PrototypeNet(args.nclass, args.nbit)
    prototype_net.cuda()

    # 2. define loss
    criterion_cl = nn.CrossEntropyLoss().cuda()
    criterion_l2 = nn.MSELoss().cuda()

    # 3. define optimizer
    optimizer = torch.optim.Adam(prototype_net.parameters(), args.lr)

    for epoch in range(args.num_epoch):  # epoch start 0
        correct = 0

        # 4. train model
        prototype_net.train()

        for step, (x, y, index) in enumerate(loader): # loader return: images, wlabels, index
            optimizer.zero_grad()

            x = x.cuda().to(torch.float)
            y = y.cuda().to(torch.float) # batchsize * 80
            y_ = torch.where(y)[1]       # batchsize * 1

            # get predict label
            f1, f2, f3, h, pred_y = prototype_net(x) # 4096, 2048, 1024, nbit, nclass

            # for acc
            preds = pred_y.max(1, keepdim=True)[1]
            correct += preds.eq(y_.view_as(preds)).sum()

            # loss1: classify loss
            loss_cl = criterion_cl(pred_y, y_) * args.lamda1

            # loss2: semantic maintain loss
            s = y.mm(y.t())
            h_norm = func.normalize(h)
            temp = h_norm.mm(h_norm.t())
            temp[temp < 0] = 0
            loss_cons = criterion_l2(temp, args.k * s) * args.lamda2

            # loss3 & loss4: balance and uncorrelated
            loss_bal = torch.sum(h) / h.size(0) * args.lamda3
            loss_cor = torch.norm(h.t().mm(h), 2) * args.lamda4

            # total loss
            loss = loss_cl + loss_cons + loss_bal + loss_cor

            loss.backward()
            optimizer.step()

        print('Epoch [%3d/%3d]: Total Loss: %.6f, loss_cl: %.6f, loss_cons %.6f, loss_bal %.8f, loss_cor %.8f.' % (
            epoch+1, args.num_epoch, loss.item(), loss_cl.item(), loss_cons.item(), loss_bal.item(), loss_cor.item()))

        acc_train = float(correct) * 100. / pred_y.size(0)
        print('Accuracy: {}/{} ({:.2f}%)'.format(correct, pred_y.size(0), acc_train))

    # save
    # (1) model save : pth
    save_path = 'checkpoints/' + args.dataset + '/web_prototype_net_'+ str(args.nbit) + 'bit_epoch' + str(epoch + 1) + '.pth'
    torch.save(prototype_net, save_path)

    # (2)get & save prototype
    _, _Prototypes = get_Corrected_Labels_km(y_, f1, f2, f3, h, args.nclass)
    save_path = 'checkpoints/' + args.dataset + '/proto_' + str(args.nbit) + 'bit.pth'
    torch.save(_Prototypes, save_path)

    # (3) save the parameters of model
    save_path = 'checkpoints/' + args.dataset + '/' + str(args.nbit)
    torch.save(prototype_net.encoderIMG_1.state_dict(), os.path.join(save_path, 'PN_1.pth'))
    torch.save(prototype_net.encoderIMG_2.state_dict(), os.path.join(save_path, 'PN_2.pth'))
    torch.save(prototype_net.encoderIMG_3.state_dict(), os.path.join(save_path, 'PN_3.pth'))
    torch.save(prototype_net.hash.state_dict(), os.path.join(save_path, 'PN_h.pth'))

    print("I'm ok")

    return prototype_net

def get_Corrected_Labels_km(Y_index, F1, F2, F3, H, nclass):
    Y_index = Y_index.detach().cpu().numpy()
    F1 = F1.detach().cpu().numpy()
    # F2 = F2.detach().cpu().numpy()
    # F3 = F3.detach().cpu().numpy()
    # H = H.detach().cpu().numpy()

    # step1: get prototype
    F1_prototypes = []
    # F2_prototypes = []
    # F3_prototypes = []
    # H_prototypes  = []

    for i in range(nclass):  # i start from 0
        # F1 80*4096
        F1_samples = F1[Y_index == i]
        F1_cluster = KMeans(1, random_state=10).fit(F1_samples)
        F1_centers = F1_cluster.cluster_centers_
        F1_prototypes.append(F1_centers)

        # # F2 80 * 2048
        # F2_samples = F2[Y_index == i]
        # F2_cluster = KMeans(1, random_state=10).fit(F2_samples)
        # F2_centers = F2_cluster.cluster_centers_
        # F2_prototypes.append(F2_centers)
        #
        # # F3 80 * 1024
        # F3_samples = F3[Y_index == i]
        # F3_cluster = KMeans(1, random_state=10).fit(F3_samples)
        # F3_centers = F3_cluster.cluster_centers_
        # F3_prototypes.append(F3_centers)
        #
        # # H 80 * bit
        # H_samples = H[Y_index == i]
        # H_cluster = KMeans(1, random_state=10).fit(H_samples)
        # H_centers = H_cluster.cluster_centers_
        # H_prototypes.append(H_centers)

    F1_prototypes = np.concatenate(F1_prototypes)
    # F2_prototypes = np.concatenate(F2_prototypes)
    # F3_prototypes = np.concatenate(F3_prototypes)
    # H_prototypes  = np.concatenate(H_prototypes)

    _prototypes = {
        'F1_prototype': torch.Tensor(F1_prototypes),
        # 'F2_prototype': torch.Tensor(F2_prototypes),
        # 'F3_prototype': torch.Tensor(F3_prototypes),
        # 'H_prototype': torch.Tensor(H_prototypes)
    }

    # get refined label
    # print("calculate y_pseudo1.")
    S1 = cos_similarity(F1, F1_prototypes)
    Y_pseudo1 = np.argmax(S1, axis=1)

    # # print("calculate y_pseudo2.")
    # S2 = cos_similarity(F2, F2_prototypes)
    # Y_pseudo2 = np.argmax(S2, axis=1)
    #
    # # print("calculate y_pseudo3.")
    # S3 = cos_similarity(F3, F3_prototypes)
    # Y_pseudo3 = np.argmax(S3, axis=1)
    #
    # # print("calculate y_pseudoH.")
    # SH = cos_similarity(H, H_prototypes)
    # Y_pseudoH = np.argmax(SH, axis=1)

    _y_pseudo = {
        'y_pseudo1': torch.Tensor(Y_pseudo1),
        # 'y_pseudo2': torch.Tensor(Y_pseudo2),
        # 'y_pseudo3': torch.Tensor(Y_pseudo3),
        # 'y_pseudoh': torch.Tensor(Y_pseudoH)
    }

    return _y_pseudo, _prototypes