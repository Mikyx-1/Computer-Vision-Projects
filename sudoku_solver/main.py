import torch
from dataset import SudokuDataset, Sudoku
from model import SudokuSolver
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


class SudokuTrainer:
    def __init__(self, solver, dataset, batch_size=2, lr=1e-3, num_epochs=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.solver = solver.to(self.device)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.Adam(self.solver.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_masked = 0
            for input_board, target, mask in self.dataloader:
                input_board, target, mask = (
                    input_board.to(self.device),
                    target.to(self.device),
                    mask.to(self.device),
                )
                self.optimizer.zero_grad()
                output = self.solver(input_board)
                loss_all = self.criterion(output, target)
                loss = loss_all[mask].mean() if mask.sum() > 0 else loss_all.mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct = (pred == target) & mask
                total_correct += correct.sum().item()
                total_masked += mask.sum().item()
            self.log(epoch, total_loss, total_correct, total_masked)

    def log(self, epoch, total_loss, total_correct, total_masked):
        accuracy = total_correct / total_masked if total_masked > 0 else 0
        print(
            f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(self.dataloader):.4f}, Accuracy: {accuracy:.4f}"
        )

    def save_weights(self, path="sudoku_solver.pth"):
        torch.save(self.solver.state_dict(), path)

    def evaluate(self):
        pass  # Can be implemented for evaluation on a test set

    @staticmethod
    @torch.no_grad()
    def calculate_accuracy(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> float:
        if pred.shape[1] > 1:
            # In case, output not processed
            pred = pred.argmax(dim=1)

        correct = (pred == target) & mask
        total_correct += correct.sum().item()
        total_masked += mask.sum().item()

        accuracy = total_correct / total_masked if total_masked > 0 else 0

        return accuracy


# Initialize and train
solver = SudokuSolver(embed_dim=128)
dataset = SudokuDataset(num_samples=100, num_mask=2)
trainer = SudokuTrainer(solver, dataset)
trainer.train()
trainer.save_weights()
