
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.amp import autocast
from tqdm import tqdm
import numpy as np
import torch

def compute_metrics(pred_list, target_list):
    pred_list   = pred_list.detach().cpu().numpy()
    target_list = target_list.detach().cpu().numpy()

    return {
        "accuracy": accuracy_score(target_list, pred_list),
        "confusion": confusion_matrix(target_list, pred_list, normalize='true'),
        "f1_micro": f1_score(target_list, pred_list, average="micro"),
        "f1_macro": f1_score(target_list, pred_list, average="macro"),
        "f1_weighted": f1_score(target_list, pred_list, average="weighted"),
    }


def metrics_summary(metircs):
    acc = metircs['accuracy']
    f1_micro = metircs['f1_micro']
    f1_macro = metircs['f1_macro']
    f1_weighted = metircs['f1_weighted']
    confusion = metircs['confusion']

    print(f"===== [Metrics Summary] =====")
    print(f"Valid Acc: {acc*100:.2f}%")
    print(f"F1_micro: {f1_micro:.4f}")
    print(f"F1_macro: {f1_macro:.4f}")
    print(f"F1_weighted: {f1_weighted:.4f}")
    print(f"Confusion:\n{np.around(confusion, 3)*100}")
    print("\n")


def BERT_evaluation(model, dataloader, device):
    model.eval()
    model.to(device)

    pred_list = []
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets).logits

            preds       = torch.argmax(outputs, dim=1)
            pred_list   += [preds]
            target_list += [targets]

    pred_list   = torch.cat(pred_list,   dim=0)
    target_list = torch.cat(target_list, dim=0)
    return pred_list, target_list


def Ensemble_evaluation(model, val_loader, device, df):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for inputs in tqdm(val_loader, desc="Validation"):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            ids = inputs['ID']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            predictions = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            df.loc[ids, 'pred'] = predictions.cpu().numpy()
    df['pred'] = df['pred'].astype(int)