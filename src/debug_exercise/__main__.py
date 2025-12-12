import argparse
from .linear_gd import run

def main():
    p = argparse.ArgumentParser(description="Minimal GD debug exercise")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run(epochs=args.epochs, lr=args.lr, seed=args.seed)

if __name__ == "__main__":
    main()

