import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class ClassWeighted_Smooth_Focal(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, num_classes=4):
        super().__init__()
        self.alpha  = alpha # or class-weight
        self.gamma  = gamma
        self.eps    = label_smoothing
        self.num_classes = num_classes

        if isinstance(alpha, torch.Tensor):
            self.alpha.unsqueeze_(dim=0)

    def forward(self, logits, target):
        p_hat       = F.softmax(logits, dim=-1)
        log_p_hat   = F.log_softmax(logits, dim=-1)
        p_t         = F.one_hot(target, num_classes=self.num_classes).float()
        p_t         = (1-self.eps) * p_t + self.eps / self.num_classes

        loss        = -1 * self.alpha * p_t * (1-p_hat)**self.gamma * log_p_hat
        loss        = loss.sum(dim=-1).mean()
        return loss

def get_criterion(config, train_data, device):

    classes = np.array(config['model']['class_label'])
    class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=train_data["label"])
    class_weight = torch.from_numpy(class_weight)
    class_weight = class_weight.to(device=device, dtype=torch.float)

    loss_config = config['loss']

    if loss_config['ce']['use']:
        print("손실 함수 적용: CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    if loss_config['class_weight_ce']['use']:
        print(f"손실 함수 적용: Class Weight CE, Class weight: {class_weight}")
        return nn.CrossEntropyLoss(weight = class_weight)
    
    if loss_config['smooth_ce']['use']:
        smoothing = loss_config['smooth_ce']['smooth']
        print(f"손실 함수 적용: Smooth CE, smoothing: {smoothing}")
        return nn.CrossEntropyLoss(label_smoothing = smoothing)
    
    if loss_config['class_weight_smooth_ce']['use']:
        smoothing = loss_config['class_weight_smooth_ce']['smooth']
        print(f"손실 함수 적용: Class Weight Smooth CE, Class weight: {class_weight}, smoothing: {smoothing}")
        return nn.CrossEntropyLoss(weight = class_weight, label_smoothing = smoothing)
        
    if loss_config['focal']['use']:
        alpha = loss_config['focal']['alpha']
        gamma = loss_config['focal']['gamma']
        print(f"손실 함수 적용: Focal, alpha={alpha}, gamma={gamma}")
        return ClassWeighted_Smooth_Focal(alpha=alpha, gamma=gamma, label_smoothing=0.0, num_classes=len(classes))
        
    if loss_config['smooth_focal']['use']:
        alpha = loss_config['smooth_focal']['alpha']
        gamma = loss_config['smooth_focal']['gamma']
        smoothing = loss_config['smooth_focal']['smooth']
        print(f"손실 함수 적용: Smooth Focal, alpha={alpha}, gamma={gamma}, smoothing: {smoothing}")
        return ClassWeighted_Smooth_Focal(alpha=alpha, gamma=gamma, label_smoothing=smoothing, num_classes=len(classes))

    if loss_config['focal_class_weight']['use']:
        gamma = loss_config['focal_class_weight']['gamma']
        print(f"손실 함수 적용: Focal & Class Weight, Class weight: {class_weight}, gamma={gamma}")
        return ClassWeighted_Smooth_Focal(alpha=class_weight, gamma=gamma, label_smoothing=0.0, num_classes=len(classes))
        
    if loss_config['smooth_focal_class_weight']['use']:
        gamma = loss_config['smooth_focal_class_weight']['gamma']
        smoothing = loss_config['smooth_focal_class_weight']['smooth']
        print(f"손실 함수 적용: Smooth Focal & Class Weight, Class weight: {class_weight}, gamma={gamma}, smoothing: {smoothing}")
        return ClassWeighted_Smooth_Focal(alpha=class_weight, gamma=gamma, label_smoothing=smoothing, num_classes=len(classes))
