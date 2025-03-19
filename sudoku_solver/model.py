import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time


class SudokuSolver(nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super(SudokuSolver, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embed_dim)

        self.conv1 = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 9, kernel_size=1)  # Final output layer

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)

        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # Shape: (batch_size, 9, 9, embed_dim)
        x = x.permute(0, 3, 1, 2)  # Shape: (batch_size, embed_dim, 9, 9)

        x1 = self.softplus(self.bn1(self.conv1(x)))
        x2 = self.softplus(self.bn2(self.conv2(x1))) + x1  # Residual connection
        x3 = self.softplus(self.bn3(self.conv3(x2))) + x2  # Residual connection
        logits = self.conv4(x3)  # Output shape: (batch_size, 9, 9, 9)

        return logits


if __name__ == "__main__":
    model = SudokuSolver(embed_dim=64)
    sample = torch.randint(0, 10, (1, 9, 9))
    pred = model(sample)
    print(f"Sample: {sample}, pred: {pred.shape}")
