"""
Checklist:
- Update output model name
- Update run name
- Update checkpoint path
"""
import os
import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torchvision import transforms, utils


class Encoder(nn.Module):
    """
    Encoder for 64x64 images producing 16x16 latent feature maps.
    """
    def __init__(self, in_channels=3, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Output spatial size: 64 -> 32 -> 16 (downsample twice by stride=2)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1)  # 64->32
        self.conv2 = nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1)  # 32->16
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """
    Decoder for 16x16 latent feature maps producing 64x64 images.
    """
    def __init__(self, embedding_dim=256, out_channels=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Upsample twice by stride=2 transpose conv
        self.conv_trans1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)  # 16->32
        self.conv_trans2 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)  # 32->64
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        x = torch.sigmoid(x)
        return x


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.
    """
    def __init__(self, embedding_dim=256, num_embeddings=2048, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(embedding_dim, num_embeddings, dtype=torch.float32)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings, dtype=torch.float32))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, inputs):
        """
        inputs: (B, D, H, W)
        returns: quantized (B, D, H, W), vq_loss, perplexity, encodings
        """
        inputs_f32 = inputs.float()
        inputs_perm = inputs_f32.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat_input = inputs_perm.view(-1, self.embedding_dim)  # (B*H*W, D)
        flat_input = flat_input.clamp(-10.0, 10.0)
        emb = self.embedding

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(emb ** 2, dim=0, keepdim=True)
            - 2.0 * flat_input.matmul(emb)
        )

        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)  # (B*H*W, num_embeddings)

        quantized = encodings.float().matmul(emb.t())
        quantized = quantized.view(inputs_perm.shape).permute(0, 3, 1, 2).contiguous()
        quantized = quantized.to(inputs.dtype)

        if self.training:
            with torch.no_grad():
                enc_f = encodings.float()
                # EMA cluster sizes
                batch_cluster = enc_f.sum(0)
                self.cluster_size.mul_(self.decay).add_(batch_cluster, alpha=1.0 - self.decay)
                # EMA embedding sums
                embed_sum = flat_input.t().matmul(enc_f)  # (D, K)
                self.embedding_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)
                # Normalize with Laplace smoothing
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                new_emb = self.embedding_avg / cluster_size.unsqueeze(0)
                self.embedding.copy_(new_emb)
                # Reseed dead codes from current batch encoder outputs
                dead_mask = (batch_cluster == 0)
                if dead_mask.any():
                    dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)
                    num_dead = dead_indices.numel()
                    if flat_input.numel() > 0:
                        rand_ids = torch.randint(0, flat_input.shape[0], (num_dead,), device=flat_input.device)
                        self.embedding[:, dead_indices] = flat_input[rand_ids].t()

                self.embedding.copy_(torch.nan_to_num(self.embedding, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-3.0, 3.0))
                self.embedding_avg.copy_(torch.nan_to_num(self.embedding_avg, nan=0.0))
                self.cluster_size.copy_(torch.nan_to_num(self.cluster_size, nan=0.0))

                        

        inputs_4loss = inputs_f32
        e_latent_loss = F.mse_loss(quantized.detach().float(), inputs_4loss)
        q_latent_loss = F.mse_loss(quantized.float(), inputs_4loss.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized.to(inputs.dtype) - inputs).detach()

        avg_probs = torch.mean(encodings.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, embedding_dim=256, num_embeddings=2048, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(3, embedding_dim)
        self.pre_vq_conv = nn.Identity()  # no extra conv before quantization
        self.vq_vae = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder = Decoder(embedding_dim, 3)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_vq_conv(z_e)
        quantized, vq_loss, perplexity, encoding_indices = self.vq_vae(z_e)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity, encoding_indices
    
    def decode_code(self, code_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete code indices back into an RGB image.
        code_indices: (B, H, W) or (H, W)
        Returns (B, 3, 64, 64) in [0,1]
        """
        if code_indices.dim() == 2:
            code_indices = code_indices.unsqueeze(0)
        elif code_indices.dim() == 3 and code_indices.size(0) == 1:
            # ensure consistent dtype and batch dimension
            code_indices = code_indices.clone()
        else:
            raise ValueError(f"Unexpected shape for code_indices: {tuple(code_indices.shape)}")

        code_indices = code_indices.long()  # ensure integer indices
        emb = self.vq_vae.embedding  # (D, K)
        # F.embedding expects (...,) and adds trailing dimension D
        z_q = F.embedding(code_indices, emb.t())  # (B, H, W, D)
        if z_q.dim() == 3:  # safety fallback if batch dim squeezed
            z_q = z_q.unsqueeze(0)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        return self.decoder(z_q)


class FlatFolderDataset(Dataset):
    """
    Dataset that recursively loads all PNG/JPG images from a folder (flat folder).
    """
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.image_size = image_size
        self.paths = []
        for ext in ('**/*.png', '**/*.jpg', '**/*.jpeg'):
            self.paths.extend(glob.glob(os.path.join(root_dir, ext), recursive=True))
        self.paths = sorted(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img


def save_reconstructions(x, x_recon, epoch, save_dir):
    """
    Save an 8x2 grid of original and reconstructed images side by side.
    """
    n = min(8, x.size(0))
    comparison = torch.cat([x[:n], x_recon[:n]])
    grid = utils.make_grid(comparison, nrow=n)
    utils.save_image(grid, os.path.join(save_dir, f'recon_epoch_{epoch}.png'))


