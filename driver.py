from utils.plotting import *
from DDPM import DDPM
from utils.constraints import SimpleConstraintProjector
from datasets import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

import argparse

def main(args):
    # Your training logic goes here
    print(f"Training model with the following parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Model save path: {args.model_save_path}")
    print(f"  Use GPU: {args.use_gpu}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Problem: {args.problem}")

    if args.problem == 'bunny':
        dataset = BunnyDataset(num_samples=10000, example=True, bunny_path='data/stanford-bunny.obj', noise_level=0.02)
        data_points = torch.stack([dataset[i] for i in range(len(dataset))])


    wandb.init() # Initialize Weights and Biases logging

    # Create dataset
    dataset = SmileyFaceDataset(num_samples=10000, A=A, b=b, example=False) # torch.Dataset object
    data_points = torch.stack([dataset[i] for i in range(len(dataset))]) # data tensor

    # Create trainer
    trainer = DDPM(data_points, project_x0=False, project_x0_sample=True, constraints_dict={'linear_equality': (A,b)}, batch_size = args.batch_size, lr = args.lr, hidden_dim=args.hidden_dim, time_embed_dim = args.time_embed_dim)
    trainer.train(epochs=args.epochs) # Diffusion model training!

    checkpoint_path = f"{args.model_save_path}_epoch_{epoch+1}.pth"
    torch.save({
        "model_state_dict": trainer.denoiser.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }, checkpoint_path)

    # Log the checkpoint to wandb
    wandb.save(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Example training loop (pseudo-code)
    # for epoch in range(args.epochs):
    #     train_model(args.data_dir, args.batch_size, args.learning_rate)
    #     if args.verbose:
    #         print(f"Epoch {epoch+1}/{args.epochs} completed.")

    # Save the model (pseudo-code)
    # save_model(args.model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for a machine learning model.")

    # Required arguments
    parser.add_argument("--data_dir", type=str, default='data/', help="Path to the directory containing the training data.")

    # Optional arguments
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10).")
    parser.add_argument("--model_save_path", type=str, default="model.pth", help="Path to save the trained model (default: model.pth).")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available (default: False).")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity (default: False).")
    parser.add_argument("--problem", type=str, default='bunny', help="Built-in problem options. Currently bunny mesh is the only support.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of diffusion model (default: 128).")
    parser.add_argument("--time_embed_dim", type=int, default=32, help="Hidden dimension of time embedding in diffusion model (default: 32).")
    parser.add_argument("--trainer", type=str, default="SBDM", help="Type of diffusion model training approach (default: SBDM). Options: DDPM, SBDM, PIDM.")
    parser.add_argument("--sampler", type=str, default="DDIM", help="Type of diffusion model sampling approach (default: DDIM). Options: DDIM, PDM. Unless specified, traditional sampling will be used.")
    args = parser.parse_args()

    main(args)