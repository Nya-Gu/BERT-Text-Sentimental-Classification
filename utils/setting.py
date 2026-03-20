import torch
import wandb

def set_seed(random_seed):
    torch.manual_seed(random_seed) 
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wandb_start(config):
    project = config['wandb']['project']
    notes   = config['wandb']['notes']
    name    = config['wandb']['name']

    record_config  = {
        "model":            config['model'],
        "train":            config['train'],
        "optimizer":        config['optimizer'],
        "scheduler":        config['scheduler'],
        "loss":             config['loss'],
        "tokenizer":        config['tokenizer'],
        "seed":             config['seed'],
        }
    
    run = wandb.init(project = project,
                     notes = notes,
                     name = name,
                     config = record_config)
    
    return run