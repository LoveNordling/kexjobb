from torch.nn import Conv2d, Sigmoid
from torch import nn
from res_connections import residual_block, residual_block_big, residual_latent_block

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        input_size = channels
        self.encoder = nn.Sequential(
            # channels, 256, 256
            residual_block(input_size, 32), # 32, 64, 64
            nn.ReLU(True),
            residual_block_big(32, 64), # 64, 4, 4
            nn.ReLU(True),
            residual_latent_block(64, 32, 16, 4),
            nn.ReLU(True),
        )
        self.conv1 = Conv2d(16, 1, 1, 1)
        self.sig = Sigmoid()


    def forward(self, x):
        x = self.encoder(x)
        x = self.conv1(x)
        x = self.sig(x)
        x = x.view(-1)
        return x

