import torch
from torchsummary import summary
from torch import nn 
import numpy
class EasyNet(nn.Module):

    def __init__(self, input_shape=(100, 100, 100)):
        
        super(EasyNet, self).__init__()
        
        self.easy = nn.Sequential(
            #nn.Conv3d(2, 4, 3),
            #nn.MaxPool3d(2),
            nn.Conv3d(1, 8, 5, 2, padding=1),#24*24*24*8
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 9),#16*16*16*16
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            # nn.Conv3d(16, 16, 9),
            # nn.LeakyReLU(),
            # nn.BatchNorm3d(16)
        )
        num = pow(16,4)
        self.FC_layer = nn.Sequential(
            nn.Linear(65536,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = nn.MaxPool3d(2)(x)
        x = self.easy(x)
        x = x.view(x.size(1),-1)
        x = self.FC_layer(x) 
        #x = x.view(x.shape[0],-1)
        
        return x

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EasyNet()
    summary(model.cuda(), (1, 100, 100, 100))
