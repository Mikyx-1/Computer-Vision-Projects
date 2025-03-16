import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time


class Sudoku:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=int)

    @staticmethod
    def is_valid(board, row, col, num):
        """Check if num is not already in the row, column, or 3x3 sub-grid."""
        if num in board[row, :]:
            return False
        if num in board[:, col]:
            return False

        start_row, start_col = row - row % 3, col - col % 3
        if num in board[start_row : start_row + 3, start_col : start_col + 3]:
            return False

        return True

    def solve_board(board):
        """Backtracking solver to fill the board completely."""
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)  # Randomize choices for variety
                    for num in nums:
                        if Sudoku.is_valid(board, row, col, num):
                            board[row][col] = num
                            if Sudoku.solve_board(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    @staticmethod
    def generate_solved_board():
        """Generates and returns a completely solved Sudoku board."""
        board = np.zeros((9, 9), dtype=int)
        Sudoku.solve_board(board)
        return board


class SudokuDataset(Dataset):
    def __init__(self, num_samples, num_mask=45):
        self.num_samples = num_samples
        self.num_mask = num_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # For demonstration, use the fixed solved board.
        solved = np.array(Sudoku.generate_solved_board()) - 1
        solved = np.where(solved == 0, 1, solved)
        input_board = solved.copy()
        # Create a boolean mask to mark masked positions
        mask = np.zeros((9, 9), dtype=bool)
        # Randomly choose positions to mask (set as the special token "9")
        indices = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(indices)
        for i, j in indices[: self.num_mask]:
            input_board[i, j] = 9  # 9 represents a masked cell
            mask[i, j] = True
        # Convert to torch tensors.
        input_board = torch.tensor(input_board, dtype=torch.long)
        target = torch.tensor(
            solved, dtype=torch.long
        )  # Ground truth remains unchanged.
        mask = torch.tensor(mask, dtype=torch.bool)
        return input_board, target, mask


class SudokuSolver(nn.Module):
    def __init__(self, embed_dim=32):
        super(SudokuSolver, self).__init__()
        # There are 10 tokens: 0-8 for digits and 9 for the mask.
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embed_dim)
        # Convolutional layers to learn spatial relationships.
        self.conv1 = nn.Conv2d(
            in_channels=embed_dim, out_channels=256, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        # Final layer: produce logits for 9 classes.
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=9, kernel_size=1)

        self.softplus = nn.Softplus()

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        # x: shape (batch_size, 9, 9) with integer tokens.
        # Apply embedding: output shape becomes (batch_size, 9, 9, embed_dim)
        x = self.embedding(x)
        # Rearrange to (batch_size, embed_dim, 9, 9) for convolution.
        x = x.permute(0, 3, 1, 2)
        x = self.softplus(self.bn1(self.conv1(x)))
        x = self.softplus(self.bn2(self.conv2(x)))
        logits = self.conv3(
            x
        )  # Output shape: (batch_size, 9, 9, 9) in channels-first format: (B, 9, 9, 9) actually is (B, 9, 9, 9) with channels=9.
        # For PyTorch's CrossEntropyLoss, expected input shape is (B, C, H, W).
        return logits


# Set up device, model, optimizer, and loss function.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuSolver(embed_dim=128).to(device)
# Use reduction='none' so we can apply our mask manually.
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create dataset and DataLoader.
dataset = SudokuDataset(num_samples=100, num_mask=2)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 10000

for epoch in range(num_epochs):
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for input_board, target, mask in dataloader:
        input_board = input_board.to(device)  # Move data to GPU
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(input_board)

        # Compute loss
        loss_all = criterion(output, target)
        loss = loss_all[mask].mean() if mask.sum() > 0 else loss_all.mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy for masked positions
        pred = output.argmax(dim=1)
        correct = (pred == target) & mask
        total_correct += correct.sum().item()
        total_masked += mask.sum().item()

    accuracy = total_correct / total_masked if total_masked > 0 else 0
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}"
    )
