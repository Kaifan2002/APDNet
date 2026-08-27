import yaml
from losses.loss_manager import LossManager
from losses.loss_registry import LOSS_REGISTRY

def build_loss_manager_from_yaml(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    manager = LossManager()
    for name, entry in cfg['losses'].items():
        fn = LOSS_REGISTRY[entry['type']]
        weight = entry.get('weight', 1.0)
        manager.add_loss(name, fn, weight)
    return manager