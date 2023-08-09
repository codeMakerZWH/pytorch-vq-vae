import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Codebook, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices):
        return self.embedding(indices)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kh, kw = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kh // 2, kw // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kh // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = MaskedConv2d('B', in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = MaskedConv2d('B', out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out + x

class PixelCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, num_layers, num_embeddings, embedding_dim):
        super(PixelCNN, self).__init__()
        self.initial_conv = MaskedConv2d('A', in_channels, out_channels, kernel_size=7, padding=3)

        layers = []
        for _ in range(num_blocks):
            for _ in range(num_layers):
                layers.append(ResidualBlock(out_channels, out_channels, kernel_size=3, padding=1))
        self.residual_blocks = nn.Sequential(*layers)

        self.final_conv1 = nn.Conv2d(out_channels, 64, kernel_size=1)
        self.final_conv2 = nn.Conv2d(64, in_channels, kernel_size=1)

        # # Define the codebook for quantization
        # self.codebook = Codebook(num_embeddings, embedding_dim)

    def forward(self, x):
        out = F.relu(self.initial_conv(x))
        out = self.residual_blocks(out)
        out = F.relu(self.final_conv1(out))
        # out = self.final_conv2(out)
        # out = F.interpolate(out, size=(32, 32), mode='nearest')


        return out


    # codebook_indices = torch.argmax(out, dim=1)
        # quantized = self.codebook(codebook_indices)
        # return quantized.permute(0, 3, 1, 2).contiguous(), codebook_indices