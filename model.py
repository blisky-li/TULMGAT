import torch
from collections import defaultdict
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

import warnings
warnings.filterwarnings("ignore")

class GAT(nn.Module):
    """
    GAT
    Setting up our GAT with torch_geometric.nn

    DATA ---> GAT_CONV1 ---> RELU ---> GAT_CONV2 ---> GLOBAL MEAN POOLING
    """

    def __init__(self,input_size=128, hidden_size=32, n_head=12, output_size=64, d = 0.25):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size, heads=n_head, dropout=d)
        self.conv2 = GATConv(hidden_size * n_head, output_size, dropout=d)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class GATmodel(nn.Module):

    """
    TULMGAT

    DATA ---> MULTI-SCALE GAT ---> CONCAT ---> BILSTM --->DROPOUT ---> LINEAR ---> OUTPUT

    """
    def __init__(self, n_class, hidden_channels2=32, input_size=128, hidden_size=32, n_head=12, output_size=64):
        super(GATmodel, self).__init__()
        self.hidden_size = hidden_channels2
        self.GAT1 = GAT(input_size, hidden_size, n_head, output_size)
        self.GAT2 = GAT(input_size, hidden_size, n_head, output_size)
        self.GAT3 = GAT(input_size, hidden_size, n_head, output_size)

        self.lstm = nn.LSTM(output_size * 3, hidden_channels2, batch_first=True, bidirectional=True)#
        self.lin = Linear(hidden_channels2 * 2, n_class)

    def forward(self, x,x2,x3, edge_index,edge_index2,edge_index3, batch,batch2,batch3):
        x = self.GAT1(x,edge_index,batch)
        x2 = self.GAT2(x2, edge_index2, batch2)
        x3 = self.GAT3(x3, edge_index3, batch3)


        x = torch.cat((x,x2),dim=1)
        x = torch.cat((x, x3), dim=1)

        h0 = torch.zeros(2, x.size(0), self.hidden_size)  # 同样考虑向前层和向后层
        c0 = torch.zeros(2, x.size(0), self.hidden_size)
        x = x.view(len(x), 1, -1)

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)

        return x