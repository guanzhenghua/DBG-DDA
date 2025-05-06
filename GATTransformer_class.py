import os

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv, GATv2Conv
from other_classes import *



class GATTransformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        # print(depth)
        self.layers = nn.ModuleList()
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, QuickFix(dim, heads, GATConv(in_channels=128, out_channels=128, heads=4, add_self_loops=False, dropout=0.1, edge_dim=1)))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))


    def forward(self, data):
        # x, edge_index = data
        #
        # for attn, ff in self.layers:
        #     x = attn(x=x, edge_index=edge_index)
        #     x = ff(x)
        #
        # return x

        x, edge_index, edge_attr = data

        for attn, ff in self.layers:
            x = attn(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = ff(x)

        return x

# if __name__ == "__main__":
#     transformer = GATTransformer(128, 4, 4, 128)
#     gat = GATConv(128, 128, heads=4, concat=False, edge_dim=1)
#     gatt = GATConv(in_channels=128, out_channels=128, heads=4, add_self_loops=False, edge_dim=1, dropout=0.1)
#     a = np.random.random(size=(894,128)).astype(np.float32)
#     aa = torch.tensor(a)
#     b = np.random.random(size=(2,776186)).astype(np.float32)
#     bb = torch.LongTensor(b)
#     c = np.random.random(size=(776186)).astype(np.float32)
#     cc = torch.tensor(c)
#     t = transformer((aa, bb, cc))
#     print(t.shape)

