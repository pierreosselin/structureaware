import torch


class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.module_list = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim)
        ])

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
