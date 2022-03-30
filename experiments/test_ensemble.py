import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging

from data_aug.utils import set_seeds
from data_aug.models import ResNet18
from data_aug.datasets import get_cifar10

from bnn_priors.third_party.calibration_error import ece

@torch.no_grad()
def test_bma(net, data_loader, samples_dir, nll_criterion=None, device=None):
  net.eval()

  ens_logits = []
  ens_nll = []

  for sample_path in tqdm(Path(samples_dir).rglob('*.pt'), leave=False):
    net.load_state_dict(torch.load(sample_path))

    all_logits = []
    all_Y = []
    all_nll = torch.tensor(0.0).to(device)
    for X, Y in tqdm(data_loader, leave=False):
      X, Y = X.to(device), Y.to(device)
      _logits = net(X)
      all_logits.append(_logits)
      all_Y.append(Y)
      if nll_criterion is not None:
        all_nll += nll_criterion(_logits, Y)
    all_logits = torch.cat(all_logits)
    all_Y = torch.cat(all_Y)

    ens_logits.append(all_logits)
    ens_nll.append(all_nll)

  ens_logits = torch.stack(ens_logits)
  ens_nll = torch.stack(ens_nll)

  ce_nll = - torch.distributions.Categorical(logits=ens_logits)\
              .log_prob(all_Y).sum(dim=-1).mean(dim=-1)

  nll = ens_nll.mean(dim=-1)

  prob_pred = ens_logits.softmax(dim=-1).mean(dim=0)

  acc = (prob_pred.argmax(dim=-1) == all_Y).sum().item() / all_Y.size(0)

  ece_val = ece(all_Y.cpu().numpy(), prob_pred.cpu().numpy(), num_bins=30)

  return { 'acc': acc, 'nll': nll, 'ce_nll': ce_nll, 'ece': ece_val }


def main(seed=None, device=0, data_dir=None, samples_dir=None, batch_size=2048):
  if data_dir is None and os.environ.get('DATADIR') is not None:
      data_dir = os.environ.get('DATADIR')

  assert Path(samples_dir).is_dir()

  torch.backends.cudnn.benchmark = True

  set_seeds(seed)
  device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

  train_data, test_data = get_cifar10(root=data_dir, seed=seed, augment=False)
  
  train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

  net = ResNet18(num_classes=10).to(device).eval()

  train_metrics = test_bma(net, train_loader, samples_dir, device=device)
  train_metrics = { f'train/{k}': v for k, v in train_metrics.items() }
  test_metrics = test_bma(net, test_loader, samples_dir, device=device)
  test_metrics = { f'test/{k}': v for k, v in test_metrics.items() }
  
  logging.info(train_metrics)
  logging.info(test_metrics)

  return train_metrics, test_metrics


def main_sweep(sweep_dir=None):
  import yaml
  import pickle

  results = []

  for d in tqdm(os.listdir(sweep_dir)):
    samples_dir = Path(sweep_dir) / d / 'samples'
    if not samples_dir.is_dir():
      continue

    logging.info(f'{samples_dir}')

    config_file = Path(sweep_dir) / d / 'config.yaml'
    with open(config_file, 'r') as f:
      config = yaml.safe_load(f)
    config['run_id'] = d

    train_metrics, test_metrics = main(samples_dir=samples_dir)

    results.append({ **config, **train_metrics, **test_metrics  })

  with open(f'{sweep_dir.split("/")[-1]}.pkl', 'wb') as f:
    pickle.dump(results, f)


if __name__ == '__main__':
  import fire

  logging.getLogger().setLevel(logging.INFO)

  fire.Fire(dict(run=main, sweep=main_sweep))
