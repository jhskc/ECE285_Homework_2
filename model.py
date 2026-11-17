from torch import nn


class Generator(nn.Module):
    """Map noise of shape (N, latent_dim, 1, 1) to 256×256 RGB images."""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            # ConvTranspose2d: in_channels, out_channels, kernel_size, stride and padding.
            nn.ConvTranspose2d(latent_dim, 1024, 4, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False) , nn.BatchNorm2d(512) , nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)  , nn.BatchNorm2d(256) , nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)  , nn.BatchNorm2d(128) , nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)   , nn.BatchNorm2d(64)  , nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1)                 ,                       nn.Tanh()    ,
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    """Score each 256×256 RGB image with a real-or-fake logit."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv2d: in_channels, out_channels, kernel_size, stride and padding.
            nn.Conv2d(3, 32, 4, 2, 1)               ,                      nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False)  , nn.BatchNorm2d(64) , nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False) , nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4)
        )

    def forward(self, images):
        return self.network(images).flatten()
