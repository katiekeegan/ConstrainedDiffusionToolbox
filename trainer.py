import wandb
from your_dataset_module import YourDataset  # Replace with your actual dataset module
from your_model_module import MLPDenoiser, SimpleConstraintProjector  # Replace with your actual model module

# Initialize wandb
wandb.init(project="ddpm-training", config={
    "timesteps": 100,
    "batch_size": 128,
    "lr": 1e-3,
    "mu": 0.01,
    "project_x0": True,
    "penalize_P": True,
    "penalize_orth_dist": True,
    "project_x0_sample": True,
    "epochs": 100,
})

# Load dataset
dataset = YourDataset()  # Replace with your actual dataset

# Initialize DDPM
ddpm = DDPM(
    dataset=dataset,
    timesteps=wandb.config.timesteps,
    batch_size=wandb.config.batch_size,
    lr=wandb.config.lr,
    mu=wandb.config.mu,
    project_x0=wandb.config.project_x0,
    penalize_P=wandb.config.penalize_P,
    penalize_orth_dist=wandb.config.penalize_orth_dist,
    project_x0_sample=wandb.config.project_x0_sample,
)

# Train the model
ddpm.train(epochs=wandb.config.epochs)

# Finish wandb run
wandb.finish()