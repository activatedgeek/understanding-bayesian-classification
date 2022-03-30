import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Dirichlet, kl_divergence


class GaussianPriorAugmentedCELoss(nn.Module):
  '''Scaled CrossEntropy + Gaussian prior + Gaussian consistency.
  '''
  def __init__(self, params, aug_scale=10, prior_scale=1, likelihood_temp=1,
               logits_temp=1):
    super().__init__()

    self.theta = params
    self.omega = aug_scale
    self.sigma = prior_scale
    self.T = likelihood_temp
    self.logits_T = logits_temp

    self.ce = nn.CrossEntropyLoss()

  def forward(self, logits, Y, logits_aug=None, N=1, K=1):
    energy = self.ce(logits / self.logits_T, Y).mul(N / self.T)
    
    if logits_aug is not None:
      p_aug = Normal(logits.unsqueeze(1), self.omega)
      energy -= p_aug.log_prob(logits_aug).sum(dim=-1).mean(dim=[-2, -1]).mul(K * N)

    for p in self.theta:
      prior = Normal(torch.zeros_like(p), self.sigma)
      energy -= prior.log_prob(p).sum()
    
    return energy


class CPriorAugmentedCELoss(nn.Module):
  '''Standard CrossEntropy + Gaussian prior + Dirichlet Logits Data Prior.
  '''
  def __init__(self, params, prior_scale=1, logits_temp=1, dir_noise=1e-4):
    super().__init__()

    self.theta = params
    self.sigma = prior_scale
    self.logits_T = logits_temp
    self.alpha_eps = dir_noise

    self.ce = nn.CrossEntropyLoss()

  def forward(self, logits, Y, N=1, diri=False):
    energy = self.ce(logits, Y).mul(N)

    if diri:
      # energy -= self.omega * (logits.div(self.logits_T).logsumexp(dim=-1) - logits.logsumexp(dim=-1).div(self.logits_T)).mean().mul(N)

      cprior = Dirichlet(torch.ones(logits.size(-1), device=logits.device) * self.alpha_eps)
      energy -= cprior.log_prob(logits.softmax(dim=-1)).mean().mul(N)

      # energy += (logits * (self.alpha_eps - 1)).mean().mul(N)

      # mix = Categorical(torch.ones(2, device=logits.device) * .5)
      # C = logits.size(-1)
      # comp = Dirichlet(torch.cat([
      #     torch.ones(1, C, device=logits.device),
      #     torch.ones(1, C, device=logits.device) * self.alpha_eps,
      # ], dim=0))
      # cprior = MixtureSameFamily(mix, comp)
      # energy -= cprior.log_prob(logits.softmax(dim=-1)).mean().mul(N)
    
    for p in self.theta:
      prior = Normal(torch.zeros_like(p), self.sigma)
      energy -= prior.log_prob(p).sum()
    
    return energy


class NoisyDirichletLoss(nn.Module):
  '''
  Get unbiased estimate by multiplying by N
  '''
  def __init__(self, params, num_classes=10, noise=1e-2, prior_scale=1,
               reduction='mean', likelihood_temp=1):
    super().__init__()

    assert noise > 0

    self.reduction = reduction

    self.theta = params
    self.C = num_classes
    self.ae = noise
    self.sigma = prior_scale
    self.T = likelihood_temp

  def forward(self, logits, Y, N=1):
    alpha = F.one_hot(Y, self.C) + self.ae
    gamma_var = (1 / alpha + 1).log()
    gamma_mean = alpha.log() - gamma_var / 2
    p_obs = Normal(gamma_mean, gamma_var.sqrt())
    energy = - p_obs.log_prob(logits).sum(dim=-1)
    if self.reduction == 'mean':
      energy = energy.mean(dim=-1).div(self.T)
    else:
      energy = energy.sum(dim=-1).div(self.T)

    for p in self.theta:
      prior = Normal(torch.zeros_like(p), self.sigma)
      energy -= prior.log_prob(p).sum().div(N)
    
    return energy


class KLAugmentedNoisyDirichletLoss(nn.Module):
  def __init__(self, params, num_classes=10, noise=1e-4, aug_scale=1, prior_scale=1,
               likelihood_temp=1):
    super().__init__()

    assert noise > 0

    self.theta = params
    self.C = num_classes
    self.ae = noise
    self.omega = aug_scale
    self.sigma = prior_scale
    self.T = likelihood_temp

  def forward(self, logits, Y, logits_aug=None, N=1, K=1):
    alpha = F.one_hot(Y, self.C) + self.ae
    gamma_var = (1 / alpha + 1).log()
    gamma_mean = alpha.log() - gamma_var / 2
    p_obs = Normal(gamma_mean, gamma_var.sqrt())
    energy = - p_obs.log_prob(logits).sum(dim=-1).mean(dim=-1).mul(N / self.T)

    if logits_aug is not None:
      p_y = Categorical(logits=logits.unsqueeze(1))
      p_y_aug = Categorical(logits=logits_aug)
      energy += kl_divergence(p_y_aug, p_y).mean(dim=[-2, -1]).mul(K * N).div(self.omega)

    for p in self.theta:
      prior = Normal(torch.zeros_like(p), self.sigma)
      energy -= prior.log_prob(p).sum()
    
    return energy


class KLAugmentedCELoss(nn.Module):
  '''Scaled CrossEntropy + Gaussian prior + KL consistency.
  '''
  def __init__(self, params, prior_scale=1, aug_scale=1):
    super().__init__()

    self.theta = params
    self.omega = aug_scale
    self.sigma = prior_scale

    self.ce = nn.CrossEntropyLoss()

  def forward(self, logits, Y, logits_aug=None, N=1, K=1):
    energy = self.ce(logits, Y).mul(N)
    
    if logits_aug is not None:
      p_y = Categorical(logits=logits.unsqueeze(1))
      p_y_aug = Categorical(logits=logits_aug)
      energy += kl_divergence(p_y_aug, p_y).mean(dim=[-2, -1]).mul(K * N).div(self.omega)

    for p in self.theta:
      prior = Normal(torch.zeros_like(p), self.sigma)
      energy -= prior.log_prob(p).sum()
    
    return energy
