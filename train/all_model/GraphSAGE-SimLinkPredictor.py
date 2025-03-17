import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from typing import Tuple

class GraphSAGE(nn.Module):
    def __init__(
            self,
            dim_in: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
            aggr: str = 'mean'):
        super().__init__()
        self.dim_in = dim_in
        self.dropout = dropout

        self.conv1 = tgnn.SAGEConv(
            in_channels=dim_in,
            out_channels=hidden_size,
            aggr=aggr
        )

        self.conv2 = tgnn.SAGEConv(
            in_channels=hidden_size,
            out_channels=output_size,
            aggr=aggr
        )

        self.dim_out = output_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = nn.functional.relu(h)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        return h


class LinkPredModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        hidden_size: int,
        output_size: int,
        hidden_dims: Tuple[int, ...] = (16, 4),
        dropout: float = 0.1,
        ):
        super().__init__()

        self.gnn = GraphSAGE(dim_in, hidden_size=hidden_size, output_size=output_size, dropout=dropout)

        self.input_dim = self.gnn.dim_out + 16

        self.in_layer = nn.Linear(self.input_dim, hidden_dims[0])
        self.hidden_layer = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out_layer = nn.Linear(hidden_dims[1], 1)

        self.lrelu = nn.LeakyReLU(0.01)

        self.dropout = dropout

        self.bn0 = nn.BatchNorm1d(self.input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, label_edge_index: torch.Tensor, label_edge_feature: torch.Tensor) -> torch.Tensor:
        h = self.gnn(x, edge_index)

        h_src = h[label_edge_index[0, :]]
        h_dst = h[label_edge_index[1, :]]

        src_dst_mult = h_src * h_dst

        all_features = torch.cat([src_dst_mult, label_edge_feature], dim=1)

        _out = self.bn0(all_features)

        _out = self.in_layer(_out)
        _out = self.bn1(_out)
        _out = self.lrelu(_out)
        _out = nn.functional.dropout(_out, p=self.dropout, training=self.training)  # 添加 Dropout

        _out = self.hidden_layer(_out)
        _out = self.bn2(_out)
        _out = self.lrelu(_out)
        _out = nn.functional.dropout(_out, p=self.dropout, training=self.training)  # 添加 Dropout

        _out = self.out_layer(_out)

        return _out