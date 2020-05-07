from pathlib import Path

if __name__ == "__main__":

    # Create project folder organization
    Path.mkdir(Path('results/logs'), parents=True, exist_ok=True)
    Path.mkdir(Path('results/images'), parents=True, exist_ok=True)
    Path.mkdir(Path('results/checkpoints'), parents=True, exist_ok=True)
