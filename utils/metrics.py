import torch
import torch.nn.functional as F

# Gaussian Kernel function
def gaussian_kernel(x, y, bandwidth=0.1):
    dist = torch.norm(x - y, dim=1)
    return torch.exp(-dist**2 / (2 * bandwidth**2))

# KDE function using Gaussian kernels
def kde(torch_points, bandwidth=0.1):
    n = torch_points.shape[0]
    kde_values = torch.zeros(n)

    for i in range(n):
        kde_values[i] = torch.mean(gaussian_kernel(torch_points[i], torch_points, bandwidth))
    
    return kde_values / torch.sum(kde_values)

# Compute KL divergence from two point clouds
def compute_kl_divergence_kde(pc1, pc2, bandwidth=0.1):
    kde_p = kde(pc1, bandwidth)
    kde_q = kde(pc2, bandwidth)

    return torch.mean(torch.log(kde_p + 1e-8) - torch.log(kde_q + 1e-8))


# Compute KL divergence using histograms in PyTorch
def compute_kl_divergence_hist(pc1, pc2, bins=20):
    # Get the min and max for each dimension to create histogram bins
    min_vals = torch.min(torch.cat((pc1, pc2), dim=0), dim=0)[0]
    max_vals = torch.max(torch.cat((pc1, pc2), dim=0), dim=0)[0]
    
    # Normalize point clouds between 0 and 1 for histogram computation
    pc1_norm = (pc1 - min_vals) / (max_vals - min_vals)
    pc2_norm = (pc2 - min_vals) / (max_vals - min_vals)

    # Get histograms in the range [0, 1] using bins
    hist1 = torch.histc(pc1_norm[:, 0], bins=bins, min=0, max=1)
    hist2 = torch.histc(pc2_norm[:, 0], bins=bins, min=0, max=1)

    # Normalize histograms
    hist1 = hist1 / torch.sum(hist1)
    hist2 = hist2 / torch.sum(hist2)

    # Compute KL divergence
    kl_div = torch.sum(hist1 * torch.log(hist1 / (hist2 + 1e-10) + 1e-10))
    return kl_div