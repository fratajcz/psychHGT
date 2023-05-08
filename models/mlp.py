import torch

class MLP(torch.nn.Module):
    def __init__(self, hidden_layers=2, in_features=-1, out_features=3, dim_hid=10, act="ReLU", dropout=0):
        self.hidden_layers = hidden_layers
        self.in_features = in_features
        self.out_features = out_features
        self.dim_hid = dim_hid
        self.act = act
        self.dropout = dropout
        layers = []
        if hidden_layers == 0:
            layers.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
        else:
            layers.append(torch.nn.Linear(in_features=in_features, out_features=dim_hid))
            layers.append(getattr(torch.nn, act)())
            layers.append(torch.nn.Dropout(p=dropout))

        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(in_features=dim_hid, out_features=dim_hid))
            layers.append(getattr(torch.nn, act)())
            layers.append(torch.nn.Dropout(p=dropout))

        if hidden_layers != 0:
            layers.append(torch.nn.Linear(in_features=dim_hid, out_features=out_features))

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.nn(x)