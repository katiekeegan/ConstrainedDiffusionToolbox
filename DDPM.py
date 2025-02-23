import torch
import wandb
from torch.utils.data import DataLoader
from torch import optim, nn

class DDPM:
    def __init__(
        self,
        dataset,
        timesteps=100,
        batch_size=128,
        lr=1e-3,
        mu=0.01,
        project_x0=True,
        penalize_P=True,
        penalize_orth_dist=True,
        project_x0_sample=True,
        constraints_dict={},
    ):
        self.project_x0 = project_x0
        self.penalize_P = penalize_P
        self.penalize_orth_dist = penalize_orth_dist
        self.project_x0_sample = project_x0_sample
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.project_x0 or self.penalize_P or self.penalize_orth_dist:
            self.projector = SimpleConstraintProjector()
            self.projector.add_constraints_from_dict(constraints_dict)
        else:
            self.projector = None
        self.training_losses = []  # Store loss for each epoch
        self.projection_norms = []  # Store projection norm for each epoch
        # Prepare dataset
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Diffusion process
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.mu = mu

        # Denoiser model
        self.denoiser = MLPDenoiser().to(self.device)
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward_diffusion(self, x0, t):
        """Adds noise to x0 at time step t"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1)

        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise  # Return noisy x and true x0

    def train(self, epochs=100):
        """Train the diffusion model."""
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            total_projection_norm = 0.0
            for x0 in self.dataloader:
                x0 = x0.to(self.device)
                t = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device)
                xt, noise = self.forward_diffusion(x0, t)
                epsilon_pred = self.denoiser(xt, t)

                # Estimate x0
                alpha_t = self.alphas[t].unsqueeze(-1)
                alpha_bars_t = self.alpha_bars[t].unsqueeze(-1)
                beta_t = self.betas[t].unsqueeze(-1)
                x0_estimate = (
                    xt - (1 - alpha_bars_t).sqrt() * epsilon_pred
                ) / alpha_bars_t.sqrt()
                if self.project_x0 or self.penalize_P:
                    x0_estimate_projected, norm_residual, normals = (
                        self.projector.project(x0_estimate)
                    )  # Apply learned projection
                    if self.project_x0:
                        loss = self.criterion(x0_estimate_projected.squeeze(), x0)
                    else:
                        loss = self.criterion(x0_estimate, x0)
                else:
                    loss = self.criterion(x0_estimate, x0)
                if self.penalize_P:
                    loss = (
                        loss
                        + self.mu
                        * torch.norm(alpha_t.unsqueeze(-1) * norm_residual) ** 2
                    )
                if self.penalize_orth_dist:
                    loss = loss + self.mu * torch.norm(
                        alpha_t.unsqueeze(-1)
                        * torch.dot(x0 - x0_estimate_projected, normal)
                    )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if self.project_x0 or self.penalize_P:
                    total_projection_norm += torch.norm(projection).item()
            if total_projection_norm != 0:
                avg_projection_norm = total_projection_norm / len(self.dataloader)
                self.projection_norms.append(avg_projection_norm)
            # Compute average loss for the epoch
            avg_loss = total_loss / len(self.dataloader)
            self.training_losses.append(avg_loss)

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "projection_norm": avg_projection_norm if total_projection_norm != 0 else 0,
            })

    def sample(self, num_samples=1000):
        """Generate 2D points using epsilon prediction with projection into the unit circle."""
        x_t = torch.randn(
            num_samples, 3, device=self.device
        )  # Start with Gaussian noise

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.float32
            )
            epsilon_pred = self.denoiser(x_t, t_tensor)

            alpha_t = self.alphas[t]  # .unsqueeze(-1).repeat(1,3)
            alpha_bar_t = self.alpha_bars[t]  # .unsqueeze(-1).repeat(1,3)
            beta_t = self.betas[t]  # .unsqueeze(-1).repeat(1,3)
            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_pred
            )
            if t > 0:
                z = torch.randn_like(x_t)
                std_dev = torch.sqrt(beta_t)
                x_t = mean + std_dev * z
            else:
                x_t = mean  # , _ = self.learnable_projection(mean)
        return x_t.cpu().detach().numpy()

    def get_training_metrics(self):
        return self.training_losses  # Retrieve stored losses for analysis