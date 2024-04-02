import torch
import torch.nn as nn


class FC(nn.Module):
    def __init__(self, dims):
        super(FC, self).__init__()

        def dense_block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f, bias=True),
                nn.ReLU(),
                nn.Dropout()
            )
        self.layers = nn.Sequential(*[dense_block(dims[i], dims[i + 1]) for i in range(len(dims) - 2)],
                                    nn.Linear(dims[-2], dims[-1], bias=True))

    def forward(self, x):
        return self.layers(x)
