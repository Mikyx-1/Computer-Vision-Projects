import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = activation()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Residual connection
        out = self.activation(out)
        return out


class SudokuSolver(nn.Module):
    def __init__(
        self, embed_dim: int = 32, num_res_blocks: int = 4, channels: int = 256
    ) -> None:
        super(SudokuSolver, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embed_dim)

        # Project the embedding to the required number of channels
        self.input_conv = nn.Conv2d(embed_dim, channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()

        # Create a stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(channels, dropout_rate=0.1, activation=nn.ReLU)
                for _ in range(num_res_blocks)
            ]
        )

        # Optional: an attention layer for capturing global dependencies (uncomment if desired)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=8, batch_first=True
        )

        # Final output layer that reduces channels to the 9 possible classes for each cell.
        self.output_conv = nn.Conv2d(channels, 9, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 9, 9)
        x = self.embedding(x)  # (batch_size, 9, 9, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (batch_size, embed_dim, 9, 9)

        x = self.activation(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # If using attention, reshape to (batch_size, num_patches, channels)
        b, c, h, w = x.shape
        x_attn = x.view(b, c, h * w).permute(0, 2, 1)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x_attn.permute(0, 2, 1).view(b, c, h, w)

        logits = self.output_conv(x)  # (batch_size, 9, 9, 9) as desired
        return logits


if __name__ == "__main__":
    model = SudokuSolver(embed_dim=64)
    sample = torch.randint(0, 10, (1, 9, 9))
    pred = model(sample)
    print(f"Sample: {sample}, pred: {pred.shape}")
