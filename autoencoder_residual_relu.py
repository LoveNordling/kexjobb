from torch import nn

from torch.nn import Conv2d, ConvTranspose2d
from res_connections import residual_block, residual_block_big, residual_latent_block, residual_block_transpose, residual_block_tranpose_big, residual_block_transpose_small, residual_latent_block_transpose






class autoencoder(nn.Module):
    def __init__(self, channels, inpaint):
        super(autoencoder, self).__init__()
        self.inpaint = inpaint
        input_size = channels
        self.encoder = nn.Sequential(
            # channels, 256, 256
            residual_block(input_size, 32), # 32, 64, 64
            nn.ReLU(True),
            residual_block_big(32, 128), # 128, 4, 4
            nn.ReLU(True),
            residual_latent_block(128, 512, 64, 4),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            residual_latent_block_transpose(64, 128, 512, 4), # 128, 4, 4
            nn.ReLU(True),
            residual_block_tranpose_big(128, 32),
            nn.ReLU(True),
            residual_block_transpose(32, input_size),  # input_size, 256, 25
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


