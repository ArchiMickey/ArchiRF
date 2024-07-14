import torch
import torch.nn as nn
import torch.nn.functional as F


from autoencoder.modules.networks_ae import Encoder, Decoder, DiagonalGaussianDistribution


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        channel_mult,
        n_res_blocks,
        z_channels,
        **kwargs
    ):
        super().__init__()
        self.encoder = Encoder(
            channels, channel_mult, n_res_blocks, in_channels, z_channels * 2
        )
        self.decoder = Decoder(
            channels, channel_mult, n_res_blocks, in_channels, z_channels
        )

    @classmethod
    def from_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        model = AutoencoderKL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
    
    def encode(self, x):
        latent_dist = self.encoder(x)
        posteriors = DiagonalGaussianDistribution(latent_dist)
        return posteriors
    
    def decode(self, posteriors):
        z = posteriors.sample()
        z = F.tanh(z)
        return self.decoder(z)
    
    def forward(self, x):
        posteriors = self.encode(x)
        x_recon = self.decode(posteriors)
        return x_recon, posteriors
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight

class Autoencoder(nn.Module):
    def __init__(self, in_channels, channels, channel_mult, n_res_blocks, z_channels, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.encoder = Encoder(channels, channel_mult, n_res_blocks, in_channels, z_channels)
        self.decoder = Decoder(channels, channel_mult, n_res_blocks, in_channels, z_channels)
    
    @classmethod
    def from_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        model = Autoencoder(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
    
    def encode(self, x):
        z = self.encoder(x)
        z = z * self.scale
        z = torch.clamp(z, -1, 1)
        return z
    
    def decode(self, z):
        z = z / self.scale
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight