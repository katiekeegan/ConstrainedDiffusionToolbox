
def PDMSampler(trainer, projector, num_samples = 1000, sigma_schedule = None, M = 10):
    if not sigma_schedule:
        sigma_schedule = torch.linspace(0.0001, 0.02, trainer.timesteps).to(trainer.device)
    x_t = torch.randn(
            num_samples, 3, device=self.device
        )  # Start with Gaussian noise
    for t in reversed(range(trainer.timesteps)):
        gamma_t = sigma_schedule[t]**2
        t_tensor = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.float32
            )
        for i in range(0, M):
            x_t_i = x_t
            eps = torch.randn(
            num_samples, 3, device=self.device
            )  # Start with Gaussian noise
            g = self.score_model(x_t, t_tensor)
            x_t_i = projector.project(x_t_i + gamma_t * g + torch.sqrt(2 * gamma_t) * eps)
        x_t = x_t_i
    return x_t

def DDIMSampler(trainer, num_samples = 1000, sigma_schedule = None, M = 10):
    if not sigma_schedule:
        sigma_schedule = torch.linspace(0.0001, 0.02, trainer.timesteps).to(trainer.device)
    x_t = torch.randn(
            num_samples, 3, device=self.device
        )  # Start with Gaussian noise
    for t in reversed(range(trainer.timesteps)):
        gamma_t = sigma_schedule[t]**2
        t_tensor = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.float32
            )
        for i in range(0, M):
            x_t_i = x_t
            eps = torch.randn(
            num_samples, 3, device=self.device
            )  # Start with Gaussian noise
            g = self.score_model(x_t, t_tensor)
            x_t_i = projector.project(x_t_i + gamma_t * g + torch.sqrt(2 * gamma_t) * eps)
        x_t = x_t_i
    return x_t
