from torch import nn

class MultiLayerPerceptronTorch(nn.Module):

    def __init__(self, bias=True):
        super(MultiLayerPerceptronTorch, self).__init__()
        self.fcnet = nn.Sequential(
            nn.Linear(1, 128, bias=bias),
            nn.Sigmoid(),
            nn.Linear(128, 256, bias=bias),
            nn.Sigmoid(),
            nn.Linear(256, 128, bias=bias),
            nn.Sigmoid(),
            nn.Linear(128, 1, bias=bias),
        )
        
    
    def forward(self, x):
        return self.fcnet(x)