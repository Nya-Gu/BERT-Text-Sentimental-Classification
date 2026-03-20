from torch.amp import autocast
import torch.nn as nn
from tqdm import tqdm
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from .tokenizer import build_custom_tokenizer
from .dataset import ReviewDataset
from .model import get_model

import wandb

def get_pretrained(model_name, config, save_path, train_data, valid_data):
    config['model']['name'] = model_name

    tokenizer       = build_custom_tokenizer(config)
    max_seq_length  = config['tokenizer']['max_length']
    valid_batch     = config['valid']['batch_size']
    valid_workers   = config['valid']['num_workers']

    train_dataset   = ReviewDataset(train_data["review"], train_data["label"], tokenizer, max_seq_length)
    valid_dataset   = ReviewDataset(valid_data["review"], valid_data["label"], tokenizer, max_seq_length)

    train_loader    = DataLoader(train_dataset,
                                batch_size  = valid_batch,
                                collate_fn  = DataCollatorWithPadding(tokenizer=tokenizer),
                                num_workers = valid_workers,
                                shuffle     = False)

    valid_loader    = DataLoader(valid_dataset,
                                 batch_size  = valid_batch,
                                 collate_fn  = DataCollatorWithPadding(tokenizer=tokenizer),
                                 num_workers = valid_workers,
                                 shuffle     = False)
    
    model_config     = config['model']
    pretrained_model = get_model(model_config, tokenizer)
    pretrained_model.load_state_dict(torch.load(save_path, map_location='cuda'))
    pretrained_model.eval()

    return pretrained_model, train_loader, valid_loader

def get_logits(model, dataloader, device):
    model.eval()
    model.to(device)

    logit_list = []
    target_list = []

    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Validation"):
            input_ids       = inputs['input_ids']
            attention_mask  = inputs['attention_mask']
            targets         = inputs['labels']

            input_ids       = input_ids.to(device)
            attention_mask  = attention_mask.to(device)
            targets         = targets.to(device)

            with autocast(device_type="cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets).logits

            logit_list  += [logits]
            target_list += [targets]

    logit_list  = torch.cat(logit_list,  dim=0)
    target_list = torch.cat(target_list, dim=0)
    return logit_list, target_list

class MLP_Classifier(nn.Module):
    def __init__(self, in_size, out_size, inner_width, depth, dropout=0.3):
        super().__init__()

        layers = []
        curr_in = in_size
        
        for i in range(depth):
            layers += [
                nn.Linear(curr_in, inner_width),
                nn.BatchNorm1d(inner_width),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ]
            curr_in = inner_width

        layers += [nn.Linear(curr_in, out_size)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

class LogitDataset(Dataset):
    def __init__(self, logits, targets):
        self.logits = logits
        self.targets = targets

    def __len__(self):
        return len(self.logits)
    
    def __getitem__(self, idx):
        return self.logits[idx], self.targets[idx]

def classifier_train(model, criterion, optimizer, train_loader, valid_loader, epochs, device, scheduler):
    model.to(device)
    model.train()

    optimizer.zero_grad()

    best_acc = 0
    print_call = len(train_loader) // 5

    for epoch in range(1, epochs+1):
        print(f"======== Epoch: {epoch} ========")
        train_loss = 0
        data_num = 0
        step = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs = inputs.to(device).float()
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.shape[0]
            data_num   += targets.shape[0]

            preds   = torch.argmax(outputs, dim=1)
            correct = (preds == targets).sum().item()

            if step % print_call == 0:
                print(f"\tTrain Loss: {train_loss / data_num:.4f}")

            current_epoch = (epoch-1) + step / len(train_loader)

            log_dict = {
                "train/loss": loss.item(),
                "train/acc": (preds == targets).sum().item() / targets.shape[0],
                "train/LR": scheduler.get_last_lr()[0],
                "epoch": current_epoch,
                }

            wandb.log(log_dict)
            scheduler.step()
            step += 1
            
        train_loss = train_loss / data_num
        val_loss, val_acc = classifier_evaluation(model, criterion, valid_loader, device, epoch)

        print(f"Epoch Summary({epoch}/{epochs}): Train Loss: {train_loss:.4f} // Val Loss: {val_loss:.4f} // Val Acc: {val_acc*100:.2f}%")

        if best_acc < val_acc:
                print(f"Best performance at epoch: {epoch}, {best_acc:.4f} -> {val_acc:.4f}\n")
                best_acc = val_acc
                torch.save(model.state_dict(), './best_classifier.pt')


def classifier_evaluation(model, criterion, valid_loader, device, epoch):
    model.eval()
    correct = 0
    data_num = 0
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc="Validation"):
            inputs = inputs.to(device).float()
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss * targets.shape[0]
            preds    = torch.argmax(outputs, dim=1)
            correct  += (preds == targets).sum().item()
            data_num += targets.shape[0]

        val_loss = val_loss / data_num
        val_acc = correct / data_num

    log_dict = {
        "val/loss": val_loss,
        "val/val acc": val_acc,
        "epoch": epoch,
        }

    wandb.log(log_dict)

    model.train()
    return val_loss, val_acc
