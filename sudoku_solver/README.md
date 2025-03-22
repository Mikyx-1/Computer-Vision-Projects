# Sudoku Solver with Deep Learning

## Overview
This project implements a deep learning-based Sudoku solver using a custom neural network architecture. The model is trained to predict missing values in Sudoku grids with varying levels of difficulty.

## Features
- **Deep Learning Model**: Trained on a dataset of Sudoku puzzles.
- **Handles Different Difficulty Levels**: Performance degrades gracefully with increased missing values.
- **Frame-Accurate Inference**: Model is optimized for accurate predictions while maintaining efficiency.
- **Supports CPU & GPU**: Runs inference on both CUDA-enabled GPUs and CPU.

---

## Dataset
The dataset consists of Sudoku puzzles with varying numbers of masked (hidden) cells. The number of masked cells determines the difficulty level:

| Masked Cells | Accuracy |
|-------------|----------|
| < 15       | ~100%    |
| <= 20      | ~95%     |
| 50         | ~60%     |

Training samples are generated dynamically, ensuring diverse puzzle variations.

---

## Model Architecture
The model is based on:
- **Embedding Dimension**: 256
- **Residual Blocks**: 12
- **Channels**: 512
- **Uses Squeeze-and-Excitation (SE) blocks** for improved feature extraction.

The model takes a **partially masked 9×9 Sudoku board** as input and predicts the missing values.

---

## Training
The model is trained using:
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW
- **Batch Size**: 64
- **Epochs**: 50+
- **Augmentation**: Random masking of different cells per puzzle

Training is conducted using PyTorch on a CUDA-enabled GPU.

### Training Script
```bash
python train.py
```

---

## Inference
Once trained, the model can solve Sudoku puzzles efficiently.

### Run Inference
```bash
python inference.py
```

### Example Code
```python
from model import SudokuSolver
from dataset import SudokuDataset
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SudokuSolver(embed_dim=256, num_res_blocks=12, channels=512)
    model.load_state_dict(torch.load("sudoku_solver.pth", map_location=device))
    model.eval()
    
    ds = SudokuDataset(num_samples=10)
    input_board, target, mask = ds[0]
    
    with torch.no_grad():
        pred = model(input_board.unsqueeze(0)).argmax(1).squeeze(0)
    
    accuracy = (pred == target).sum().item() / target.numel() * 100
    
    print(f"Input:\n{input_board}\n")
    print(f"Target:\n{target}\n")
    print(f"Predicted:\n{pred}\n")
    print(f"Accuracy: {accuracy:.2f}%")
```

---

## Performance Analysis
The model performs well on easier puzzles but struggles as the number of missing values increases. Strategies to improve performance include:
- **Data Augmentation**: Increase training variations.
- **Transformer-Based Architectures**: Experimenting with attention mechanisms.
- **Reinforcement Learning (Q-learning)**: Exploring alternative learning techniques.

---

## Future Work
- **Enhance accuracy for difficult puzzles** (e.g., 50+ masked cells).
- **Optimize inference time for real-time solving.**
- **Deploy as a web service using FastAPI and Next.js.**

---


