from model import SudokuSolver
from dataset import SudokuDataset, Sudoku
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SudokuSolver(embed_dim=256, num_res_blocks=12, channels=512)
    model.load_state_dict(torch.load("sudoku_solver/sudoku_solver.pth", map_location=device))
    model.eval()

    ds = SudokuDataset(num_samples=10)
    input_board, target, mask = ds[0]

    with torch.no_grad():
        pred = model(input_board.unsqueeze(0)).argmax(1).squeeze(0)  # Remove batch dimension

    # Compute accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total * 100

    print(f"Input: \n{input_board} \n")
    print(f"Target: \n{target} \n")
    print(f"Pred: \n{pred} \n")
    print(f"Accuracy: {accuracy:.2f}%")
