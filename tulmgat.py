import argparse
import random
import datetime
import torch
import numpy as np
from collections import defaultdict
torch.set_default_tensor_type(torch.FloatTensor)
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import GATmodel
from utils import train, test
from datasets import DataCenter
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

torch.set_default_tensor_type(torch.FloatTensor)
parser = argparse.ArgumentParser(description='pytorch version of GAT')

parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--b_sz', type=int, default=96)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--hidden_channels', type=int, default=120, help='Hidden_channels OF LSTM')
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding Size OF NODEs')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden Size OF GAT')
parser.add_argument('--output_size', type=int, default=64, help='Output Size OF GAT & Input Size OF LSTM')
parser.add_argument('--n_heads', type=int, default=12, help='N_Heads OF Multi_Head GAT')
parser.add_argument('--lr', type=float, default=0.0025, help='Learning Rate')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))
else:
    print('CPU mode')
# device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cuda")
print('DEVICE:', device)

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    seed = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_tensor_type(torch.FloatTensor)

    # load data

    dataCenter = DataCenter(None)
    dataCenter.load_dataSet(embedding_path='gowemb800new128_2.dat', trajectory_path='gowalla800.txt')
    trainset = getattr(dataCenter,'_train')
    trainset2 = getattr(dataCenter, '_train2')
    trainset3 = getattr(dataCenter, '_train3')
    testset = getattr(dataCenter, '_test')
    testset2 = getattr(dataCenter, '_test2')
    testset3 = getattr(dataCenter, '_test3')
    n_class = getattr(dataCenter,'_number')  # 329 105 149 247
    number_1 = len(trainset)
    number_2 = len(testset)

    """
    Randomised data sets.
    
    Since the randomisation of the torch_geometric.loader is difficult to set a random seed, 
    we disrupt the dataset in advance without having to randomise it later when reading the data(DataLoader)
    """


    r = random.random
    random.seed(seed)
    random.shuffle(trainset, random=r)
    random.seed(seed)
    random.shuffle(trainset2, random=r)
    random.seed(seed)
    random.shuffle(trainset3, random=r)

    """
    Read the parameters in args
    """
    b_sz = args.b_sz
    hidden_channels = args.hidden_channels
    input_size = args.embedding_size
    hidden_size = args.hidden_size
    n_head = args.n_heads
    output_size = args.output_size
    lr = args.lr

    """
    formal dataset
    """
    train_loader = DataLoader(trainset, batch_size=b_sz, shuffle=False)
    train_loader2 = DataLoader(trainset2, batch_size=b_sz, shuffle=False)
    train_loader3 = DataLoader(trainset3, batch_size=b_sz, shuffle=False)
    test_loader = DataLoader(testset, batch_size=b_sz, shuffle=False)
    test_loader2 = DataLoader(testset2, batch_size=b_sz, shuffle=False)
    test_loader3 = DataLoader(testset3, batch_size=b_sz, shuffle=False)

    train_loader = list(train_loader)
    train_loader2 = list(train_loader2)
    train_loader3 = list(train_loader3)
    test_loader = list(test_loader)
    test_loader2 = list(test_loader2)
    test_loader3 = list(test_loader3)

    # SET YOUR  MODEL

    model = GATmodel(n_class, hidden_channels2=hidden_channels, input_size=input_size, hidden_size=hidden_size, n_head=n_head, output_size=output_size)
    print(model)
    new_acc = 0
    new_acc5 = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#0.004 0.0025
    criterion = torch.nn.CrossEntropyLoss()

    # Train and Test
    for epoch in range(1, args.epochs + 1):
        print(epoch)
        loss,train_acc,tr5 = train(number_1, train_loader,train_loader2,train_loader3, model,optimizer, criterion)
        test_acc,t5,dic,lst,lst2 = test(number_2, test_loader, test_loader2, test_loader3,  model, n_class)
        if new_acc < test_acc or new_acc5 < t5:
            new_acc = test_acc
            new_acc5 = t5
            lk = []
            for j in dic.keys():
                if dic[j][2] != 0:
                    lk.append(j)
            l2 = [0]
            l = [0]
            recall = 0
            prec = 0
            for j in lk:
                recall += dic[j][0]/dic[j][2]
                if dic[j][1] != 0:
                    prec += dic[j][0]/dic[j][1]
            recall = recall/len(lk)
            prec = prec/len(lk)
            f1 = 2*(prec*recall)/(prec+recall)
            p2 = precision_score(lst2, lst, average='macro')
            r2 = recall_score(lst2, lst, average='macro')
            f1score2 = f1_score(lst2, lst, average='macro')

            print(
                f'Epoch: {epoch:03d},Loss:{loss:.4f}, Train Acc: {train_acc:.4f},Train Acc5: {tr5:.4f},Test Acc: {test_acc:.4f},Test Acc5: {t5:.4f},Precision:{prec:.4f},Recall:{recall:4f},F1值：{f1:4f}')

    #model.train()
    if args.learn_method == 'sup':
        print('GAT with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')



    endtime = datetime.datetime.now()
    print(starttime)
    print(endtime)