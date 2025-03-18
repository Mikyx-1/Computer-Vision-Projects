import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

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
        if num in board[start_row:start_row+3, start_col:start_col+3]:
            return False
        
        return True
    
    @staticmethod
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
        masked_board, mask = SudokuDataset.mask_sudoku_board(solved)
        input_board = torch.tensor(masked_board, dtype=torch.long)
        target = torch.tensor(solved, dtype=torch.long)  # Ground truth remains unchanged.
        mask = torch.tensor(mask, dtype=torch.bool)
        return input_board, target, mask
    

    @staticmethod
    def mask_sudoku_board(solved_board: np.ndarray):
        mask = np.zeros((9, 9), dtype=bool)
        masked_board = solved_board.copy()
        indices = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(indices)
        num_mask = random.randint(, b) 
        for i, j in indices[:self.num_mask]:
            masked_board[i, j] = 9
            mask[i, j] = True
            
        return masked_board, mask
    
    
if __name__ == "__main__":
    ds = SudokuDataset(num_samples=100)
    input_board, target, mask = ds[0]
    print(f"input_board: {input_board}, target: {target}, mask: {mask}")