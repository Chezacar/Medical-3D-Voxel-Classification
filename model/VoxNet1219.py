import torch
from torchsummary import summary
from torch import nn 
class VoxNet(torch.nn.Module):

    def __init__(self, num_classes, input_shape=(32, 32, 32)):
                 # weights_path=None,
                 # load_body_weights=True,
                 # load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.
        Modified in order to accept different input shapes.
        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        weights_path: str or None, optional
            Default: None
        load_body_weights: bool, optional
            Default: True
        load_head_weights: bool, optional
            Default: True
        Notes
        -----
        Weights available at: url to be added
        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNet, self).__init__()
        self.preprocess = torch.nn.Sequential(
            # nn.Conv3d(2, 4, 3),
            nn.Conv3d(1, 8, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, 5),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, 5),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
        )
        self.body = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=16,
                            out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            #torch.nn.Dropout(p=0.1),
            torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(2),
            #torch.nn.Dropout(p=0.1)
        )

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((16, 16) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(
            torch.nn.Linear(32000, 1024),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.1),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
            torch.nn.Softmax()
        )

        # if weights_path is not None:
        #    weights = torch.load(weights_path)
        #    if load_body_weights:
        #        self.body.load_state_dict(weights["body"])
        #    elif load_head_weights:
        #        self.head.load_state_dict(weights["head"])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.body(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VoxNet(2).to(DEVICE)
    summary(model, (1,32, 32, 32))
