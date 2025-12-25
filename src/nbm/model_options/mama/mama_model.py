import torch
from torch import nn

from model_options.mama.multi_scale_attention_block import MultiScaleAttentionBlock1D


class HashMemory(nn.Module):
    def __init__(self, num_slots, inDim, hash_dim=None, mappingType="hard"):
        super().__init__()
        self.num_slots = num_slots
        self.inDim = inDim
        self.mappingType = mappingType

        if hash_dim is None:
            hash_dim = inDim
        else:
            self.hash_dim = hash_dim

        # Memory slots
        self.memory = nn.Parameter(torch.randn(num_slots, hash_dim))

        # Hash projection (Eq. 6)
        self.hash_func = nn.Linear(inDim, hash_dim)

        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.hash_func.weight)

    def forward(self, z):
        # input must be flatten to [B F] before passing

        # Get hash codes (Eq. 6)
        z_hash = self.hash_func(z)
        z_hash_sign = torch.sign(z_hash - 0.51)
        z_binary = (z_hash_sign + 1) / 2  # Convert to {0, 1}

        # Memory hash codes
        mem_hash = self.hash_func(self.memory)
        mem_hash_sign = torch.sign(mem_hash - 0.51)
        mem_binary = (mem_hash_sign + 1) / 2  # Convert to {0, 1}

        # Hamming distance (Eq. 7)
        h_dist = (z_binary.unsqueeze(1) != mem_binary.unsqueeze(0)).sum(dim=2)  # [B, N]

        # mapping
        if self.mappingType == "hard":
            min_indices = h_dist.argmin(dim=1)  # [B]
            z_hat = self.memory[min_indices]  # [B, feat_dim]
        elif self.mappingType == "soft":
            weights = torch.softmax(-h_dist, dim=1)  # [B, N]
            z_hat = torch.matmul(weights, self.memory)  # [B, feat_dim]

        return z_hat


class MamaAutoencoder(nn.Module):
    def __init__(
        self,
        inTempLen,
        modelMatrix=[
            (7, 8),
            (8, 4),
            (4, 4),
        ],
        mem_slots=100,
    ):
        super().__init__()

        # Encoder
        self.encoders = nn.Sequential()
        for i, (in_channels, out_channels, *_) in enumerate(modelMatrix):
            self.encoders.append(
                MultiScaleAttentionBlock1D(in_channels, out_channels, block_type="DA")
            )

        # semantic_basis
        encoded_channels = modelMatrix[-1][1]
        self.semantic_basis = MultiScaleAttentionBlock1D(
            encoded_channels, encoded_channels, block_type="SA"
        )

        # Memory module
        hashLen = inTempLen // (2 ** len(modelMatrix)) * encoded_channels
        self.memory = HashMemory(mem_slots, hashLen)

        # Decoder
        self.decoder0_0 = MultiScaleAttentionBlock1D(
            modelMatrix[-1][1] * 2, modelMatrix[-1][1], block_type="SA"
        )
        self.decoder0_1 = MultiScaleAttentionBlock1D(
            modelMatrix[-1][1], modelMatrix[-1][0], block_type="UA"
        )

        self.decoder1_2 = MultiScaleAttentionBlock1D(  # upscale semantic_basis
            encoded_channels, modelMatrix[-2][1], block_type="UA"
        )
        self.decoder1_0 = MultiScaleAttentionBlock1D(
            modelMatrix[-2][1] * 2, modelMatrix[-2][1], block_type="SA"
        )
        self.decoder1_1 = MultiScaleAttentionBlock1D(
            modelMatrix[-2][1], modelMatrix[-2][0], block_type="UA"
        )

        self.decoder_rest = nn.Sequential()
        for i in range(-3, -len(modelMatrix) - 1, -1):
            out_channels = modelMatrix[i][0]
            in_channels = modelMatrix[i][1]
            self.decoder_rest.append(
                MultiScaleAttentionBlock1D(in_channels, out_channels, block_type="UA")
            )

    def extractSemantic(self, encoded):
        # Extract semantic features
        semantic = self.semantic_basis(encoded)
        return semantic

    def encode(self, x):
        encoded = self.encoders(x)  # [B, C, L]
        semantic = self.extractSemantic(encoded)  # [B, C, L]
        encodedMemory = self.getClosestMemory(encoded)  # [B, C, L]
        return encodedMemory, semantic

    def getClosestMemory(self, encoded):
        size = encoded.size()
        # Flatten the encoded features
        z = encoded.flatten(start_dim=1)
        # Get the closest memory
        z_hat = self.memory(z)
        return z_hat.view(size)

    def decode(self, latent):
        encodedMemory, semantic_basis = latent

        z_hat = encodedMemory

        # Concatenate z_hat and semantic_basis
        z_hat = torch.cat([z_hat, semantic_basis], dim=1)

        x = self.decoder0_0(z_hat)  # [B, C, L]
        x = self.decoder0_1(x)  # [B, C, L]

        # concat x with up-sampled semantic_basis
        up_semantic = self.decoder1_2(semantic_basis)  # [B, C', L']
        x = torch.cat([x, up_semantic], dim=1)  # [B, C + C', L']
        x = self.decoder1_0(x)
        x = self.decoder1_1(x)

        x = self.decoder_rest(x)  # [B, C, L]

        return x

    def forward(self, x):
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon
