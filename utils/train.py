import wandb
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler

def train(model, criterion, optimizer, train_loader, valid_loader, epochs, device, scheduler, accum_step=1):
    model.to(device)
    model.train()

    scaler = GradScaler()
    optimizer.zero_grad()

    best_acc = 0
    print_call = len(train_loader) // 5

    for epoch in range(1, epochs+1):
        print(f"======== Epoch: {epoch} ========")
        train_loss = 0
        data_num = 0
        step = 0
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            targets = inputs['labels']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets).logits
                loss = criterion(outputs, targets)
                loss = loss / accum_step

            scaler.scale(loss).backward()

            if (step + 1) % accum_step == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            train_loss  += loss.item() * targets.shape[0]
            data_num    += targets.shape[0]

            if step % print_call == 0:
                print(f"\tTrain Loss: {train_loss / data_num:.4f}")

            current_epoch = (epoch-1) + step / len(train_loader)

            log_dict = {
                "train/loss": loss.item(),
                "train/LR": scheduler.get_last_lr()[0],
                "epoch": current_epoch,
                }

            wandb.log(log_dict)
            step += 1

        train_loss = train_loss / data_num
        val_loss, val_acc = evaluation(model, criterion, valid_loader, device, epoch)

        print(f"Epoch Summary({epoch}/{epochs}): Train Loss: {train_loss:.4f} // Val Loss: {val_loss:.4f} // Val Acc: {val_acc*100:.2f}%")

        if best_acc < val_acc:
                print(f"Best performance at epoch: {epoch}, {best_acc:.4f} -> {val_acc:.4f}\n")
                best_acc = val_acc
                torch.save(model.state_dict(), './best_model.pt')

def evaluation(model, criterion, valid_loader, device, epoch):
    model.eval()
    correct = 0
    data_num = 0
    val_loss = 0

    with torch.no_grad():
        for inputs in tqdm(valid_loader, desc="Validation"):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            targets = inputs['labels']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets).logits
            loss = criterion(outputs, targets)

            val_loss += loss * targets.shape[0]
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
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