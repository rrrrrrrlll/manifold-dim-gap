"""
GNN model code copied from image-gnn-id-estimation-2 (commit d6fb71b):
  GNNModel          <- src/scripts/models/classifier.py
  GNNBasicBlock     <- src/scripts/models/gnns.py
  WeightedSAGEConv  <- src/scripts/models/encoders.py
Class bodies are verbatim; only the imports were merged into this file.
GNNModel.forward() returns out[:batch.batch_size], so full-batch callers set
data.batch_size = number of nodes (see mdgap.train).
"""
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, GatedGraphConv, ARMAConv


class GNNBasicBlock(nn.Module):
    """
    Basic block for graph neural network (GNN) with skip connection.
    """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            gnn_conv, 
            gnn_conv_args,
            res_connect,
            **kwargs
        ):
        """
        Initialize the GNNBasicBlock module.

        Args:
            in_channels (int): GNN nodes' input size.
            out_channels (int): GNN nodes' hidden size.
            gnn_conv (nn.Module): The GNN convolutional layer.
            gnn_conv_args (dict): Arguments for the GNN convolutional layer.
            **kwargs: Additional keyword arguments.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_connect = res_connect
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                gnn_conv(
                    in_channels=in_channels, out_channels=out_channels, **gnn_conv_args
                ),
                nn.BatchNorm1d(out_channels),
                # nn.LeakyReLU(),
                # gnn_conv(
                #     in_channels=out_channels, out_channels=out_channels, **gnn_conv_args
                # ),
                # nn.BatchNorm1d(out_channels),
            ]
        )

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass of the GNNBasicBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            torch.Tensor: Transformed graph representations.
            List[torch.Tensor]: List of attention weights if `return_attention_weights` is True, otherwise an empty list.
        """

        out = x

        attention_weights = []
        if "return_attention_weights" in kwargs:
            return_attention_weights = kwargs["return_attention_weights"]
        else:
            return_attention_weights = None

        for layer in self.layers:
            if isinstance(layer, self.gnn_conv):
                out = layer(
                    x=out,
                    edge_index=edge_index,
                    **kwargs
                )
                if return_attention_weights:
                    out, attention_weight = out
                    attention_weights.append(attention_weight)
            else:
                out = layer(out)

        if self.res_connect:
            out = out + x

        out = F.leaky_relu(out)

        # out = F.dropout(out, p=0.2)

        return out, attention_weights


class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        self.root_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def update(self, aggr_out, x):
        return self.lin(aggr_out) + self.root_lin(x)


class GNNModel(nn.Module):
    """
    Graph Neural Network (GNN).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        return config

    def __init__(
        self,
        in_features,
        hidden_size,
        out_size,
        gnn_conv,
        gnn_conv_args,
        layers=1,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        gnn_conv = getattr(sys.modules[__name__], gnn_conv)
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                # nn.Linear(in_features, hidden_size),
                GNNBasicBlock(
                    in_features, 
                    hidden_size, 
                    gnn_conv, 
                    gnn_conv_args,
                    res_connect=False
                ),
                *[
                    GNNBasicBlock(
                        hidden_size,
                        hidden_size, 
                        gnn_conv, 
                        gnn_conv_args,
                        res_connect=True
                    ) for l in range(layers-1)
                ],
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out_size)
            ]
        )

    def _step(self):
        pass

    def _eval(self):
        pass

    def forward(self, batch, return_embeddings=False):
        out = batch.x

        if self.gnn_conv in (GATConv, GATv2Conv):
            conv_fwd_args = {
                "return_attention_weights": None,
                "edge_attr": batch.edge_weight
            }
        elif self.gnn_conv in (GCNConv, GatedGraphConv, ARMAConv):
            conv_fwd_args = {
                "edge_weight": batch.edge_weight
            }
        else:
            conv_fwd_args = {}

        embeddings = None
        for layer in self.layers:
            if isinstance(layer, GNNBasicBlock):
                out, _ = layer(out, batch.edge_index, **conv_fwd_args)
                embeddings = out
            elif isinstance(layer, self.gnn_conv):
                out = layer(out, batch.edge_index, **conv_fwd_args)
            else:
                out = layer(out)

        # NeighborLoader batches put the seed nodes -- the ones this batch is
        # actually supposed to predict -- as the first batch.batch_size rows, with
        # any sampled 1-/2-hop neighbors afterward (pulled in only to support
        # message-passing into the seeds, not meant to be scored themselves).
        # This used to slice by batch.mask instead, a leftover whole-graph
        # train/test flag: sample_subgraph() only keeps edges between
        # already-sampled training nodes, so that mask was true for nearly the
        # entire local batch, not just the seeds -- scoring predictions for
        # under-supported sampled neighbors (whose own 2-hop neighborhoods were
        # never fully expanded) right alongside the real seed-node predictions.
        logits = out[:batch.batch_size]
        if return_embeddings:
            return logits, embeddings[:batch.batch_size]
        return logits
