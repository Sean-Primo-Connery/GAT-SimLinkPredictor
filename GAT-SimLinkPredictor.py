import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from typing import Tuple

class GAT(nn.Module):
    def __init__(
            self,
            dim_in: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dropout = dropout

        # 第一层 GAT: 2个注意力头
        self.conv1 = tgnn.GATv2Conv(
            in_channels=dim_in,
            out_channels=hidden_size,
            heads=2,
            dropout=self.dropout,
            residual=True
        )

        # 第二层 GAT: 1个注意力头
        self.conv2 = tgnn.GATv2Conv(
            in_channels=hidden_size * 2,  # 输入维度为 hidden_size * 2（因为第一层有 2 个头）
            out_channels=output_size,
            heads=1,
            dropout=self.dropout,
            residual=True
        )

        self.dim_out = output_size  # 输出维度为 output_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 第一层
        h = self.conv1(x, edge_index)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        # 第二层
        h = self.conv2(h, edge_index)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        return h


class LinkPredModel(nn.Module):
    def __init__(
            self,
            dim_in: int,
            gnn_hidden_size: int,
            gnn_output_size: int,
            hidden_dims: Tuple[int, ...] = (16, 4),
            dropout: float = 0.1,
    ):
        super().__init__()

        # 实例化 GAT 模型
        self.gnn = GAT(
            dim_in=dim_in,
            hidden_size=gnn_hidden_size,
            output_size=gnn_output_size,
            dropout=dropout
        )

        # 输入维度
        self.input_dim = self.gnn.dim_out + 16
        self.dropout = dropout  # 保存 dropout 概率

        # MLP 层
        self.in_layer = nn.Linear(self.input_dim, hidden_dims[0])
        self.hidden_layer = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out_layer = nn.Linear(hidden_dims[1], 1)

        # 激活函数和批归一化
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn0 = nn.BatchNorm1d(self.input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, label_edge_index: torch.Tensor, label_edge_feature: torch.Tensor) -> torch.Tensor:
        # 通过 GAT 获取节点嵌入
        h = self.gnn(x, edge_index)

        # 从节点嵌入中提取源节点和目标节点的特征
        h_src = h[label_edge_index[0, :]]
        h_dst = h[label_edge_index[1, :]]

        # 特征融合（点乘）
        src_dst_mult = h_src * h_dst

        # 将 GAT 输出与边特征拼接
        all_features = torch.cat([src_dst_mult, label_edge_feature], dim=1)

        # MLP 前向传播
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