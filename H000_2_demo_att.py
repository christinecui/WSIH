import argparse
from load_data import *
from H000_3_train_att import *

# parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nus21', choices=['coco', 'nus21'])
parser.add_argument('--nbit', type=int, default=128, choices=[16, 32, 64, 128])
parser.add_argument('--batchsize', type=int, default='128')
parser.add_argument('--lr', type=float, default='0.0002')
args = parser.parse_args()

if args.dataset == 'coco':
    args.nclass = 80
    args.num_epoch = 50
    args.lamda1 = 10
    args.lamda2 = 1
    args.lamda3 = 0.1
    # seed_setting(seed=2022)

elif args.dataset == 'nus21':
    args.nclass = 21
    args.num_epoch = 50
    args.lamda1 = 10
    args.lamda2 = 10
    args.lamda3 = 10
    # seed_setting(seed=2022)
else:
    raise Exception('No this dataset!')

def train_WISH(args):
    dataloader = get_loader(args.batchsize, args.dataset)
    train_loader = dataloader['train']

    print('start WISH stage2')
    model_H = TrainHashing(args, train_loader)
    print('end WISH stage2')

    return model_H, dataloader

def performance_eval(model, dataloader):
    # load data
    database_loader = dataloader['database']
    query_loader = dataloader['query']

    model.eval().cuda()

    re_BI, re_L, qu_BI, qu_L = compress(database_loader, query_loader, model)

    ## Save
    _dict = {
        'retrieval_B': re_BI,
        'L_db':re_L,
        'val_B': qu_BI,
        'L_te':qu_L,
    }
    sava_path = 'hashcode/HASH_' + args.dataset + '_' + str(args.nbit) + 'bits.mat'
    sio.savemat(sava_path, _dict)

    return 0

def compress(database_loader, query_loader, model):

    if args.dataset == 'coco':
        proto_F_path = './checkpoints/coco/proto_' + str(args.nbit) + 'bit.pth' # type: torch
    elif args.dataset == 'nus21':
        proto_F_path = './checkpoints/nus21/proto_' + str(args.nbit) + 'bit.pth' # type: torch

    proto_F = torch.load(proto_F_path)

    # retrieval
    re_BI = list([])
    re_L = list([])
    for _, (data_I, data_L, _) in enumerate(database_loader):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            _, _, _, code_I, _ = model(var_data_I.to(torch.float), proto_F)
            # _, _, _, code_I, _ = model(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    # query
    qu_BI = list([])
    qu_L = list([])
    for _, (data_I, data_L, _) in enumerate(query_loader):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            _, _, _, code_I, _ = model(var_data_I.to(torch.float), proto_F)
            # _, _, _, code_I, _ = model(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_L = np.array(qu_L)

    return re_BI, re_L, qu_BI, qu_L

if __name__ == '__main__':
    # train
    model_trained, loader = train_WISH(args)

    # test
    performance_eval(model_trained, loader)
    print('lamda1: %.8f, lamda2: %.8f, lamda3: %.8f' % (args.lamda1, args.lamda2, args.lamda3))
    print("******************************************")