import argparse
from P000_3_train import *
from load_data import *

# parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='nus21', choices=['coco', 'nus21'])
parser.add_argument('--nbit',     type=int, default=128)
parser.add_argument('--lr', type=float, default='0.0002')
parser.add_argument('--k', type=float, default='1')

# parser.add_argument('--lamda1', type=float, default='10')
# parser.add_argument('--lamda2', type=float, default='1')

argsP = parser.parse_args()

if argsP.dataset == 'coco':
    argsP.nclass = 80
    argsP.batchsize = 8000
    argsP.num_epoch = 180
    argsP.lamda1 = 10
    argsP.lamda2 = 1
    argsP.lamda3 = 0.00001
    argsP.lamda4 = 0.00001
    # seed_setting(seed=1)
elif argsP.dataset == 'nus21':
    argsP.nclass = 21
    argsP.batchsize = 4200
    argsP.num_epoch = 120
    argsP.lamda1 = 10
    argsP.lamda2 = 0.001
    argsP.lamda3 = 0.000001
    argsP.lamda4 = 0.00001
    seed_setting(seed=1)
else:
    raise Exception('No this dataset!')

def train_WISH_proto(argsP):
    # get data
    dataloader = get_loader_web(argsP.batchsize, argsP.dataset)
    web_loader = dataloader['train']

    # train model
    model_P = getPrototype(argsP, web_loader)

    return model_P

# def performance_eval(model, dataloader):
#     # load data
#     database_loader = dataloader['database']
#     query_loader = dataloader['query']
#
#     model.eval().cuda()
#
#     re_BI, re_L, qu_BI, qu_L = compress(database_loader, query_loader, model)
#
#     ## Save
#     _dict = {
#         'retrieval_B': re_BI,
#         'L_db':re_L,
#         'val_B': qu_BI,
#         'L_te':qu_L,
#     }
#     sava_path = 'hashcode/HASH_' + argsP.dataset + '_' + str(argsP.nbit) + 'bits.mat'
#     sio.savemat(sava_path, _dict)
#
#     return 0
#
# def compress(database_loader, query_loader, model):
#
#     # retrieval
#     re_BI = list([])
#     re_L = list([])
#     for _, (data_I, data_L, _) in enumerate(database_loader):
#         with torch.no_grad():
#             var_data_I = data_I.cuda()
#             _, _, _, code_I, _ = model(var_data_I.to(torch.float))
#         code_I = torch.sign(code_I)
#         re_BI.extend(code_I.cpu().data.numpy())
#         re_L.extend(data_L.cpu().data.numpy())
#
#     # query
#     qu_BI = list([])
#     qu_L = list([])
#     for _, (data_I, data_L, _) in enumerate(query_loader):
#         with torch.no_grad():
#             var_data_I = data_I.cuda()
#             _, _, _, code_I, _ = model(var_data_I.to(torch.float))
#         code_I = torch.sign(code_I)
#         qu_BI.extend(code_I.cpu().data.numpy())
#         qu_L.extend(data_L.cpu().data.numpy())
#
#     re_BI = np.array(re_BI)
#     re_L = np.array(re_L)
#
#     qu_BI = np.array(qu_BI)
#     qu_L = np.array(qu_L)
#
#     return re_BI, re_L, qu_BI, qu_L
#
# def test_WISH(argsP, model_trained):
#     dataloader = get_loader(argsP.batchsize, argsP.dataset)
#     performance_eval(model_trained, dataloader)

if __name__ == '__main__':
    model_trained = train_WISH_proto(argsP)
    # test_WISH(argsP, model_trained)
    print('lamda1: %.8f, lamda2: %.8f, lamda3: %.8f, lamda4: %.8f' % (argsP.lamda1, argsP.lamda2, argsP.lamda3, argsP.lamda4))
    print("******************************************")
