import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

def get_optimizer(model, optim_config):
    optim_name          = optim_config['name']
    learning_rate       = float(optim_config['learning_rate'])
    weight_decay        = float(optim_config['weight_decay'])

    if optim_name == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=learning_rate,
                               weight_decay=weight_decay)
        print("옵티마이저 설정: Adam")

    elif optim_name == 'adamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        print("옵티마이저 설정: AdamW")
    else:
        assert False, "optimizer는 adam, adamW 중 하나를 선택해주시길 바랍니다."

    return optimizer


def get_scheduler(config, optimizer, batch_per_epoch):
    scheduler_list      = []

    scheduler_config    = config['scheduler']
    accum_step          = config['train']['gradient_accum']
    steps_per_epoch     = batch_per_epoch // accum_step

    warmup_epochs       = scheduler_config['linear']['warmup_epoch']
    cosine_period       = scheduler_config['cosine']['period']
    cosine_eta_min      = float(scheduler_config['cosine']['eta_min'])

    warmup_steps        = warmup_epochs * steps_per_epoch
    cosine_steps        = cosine_period * steps_per_epoch

    if scheduler_config['linear']['use']:
        scheduler_list += [LinearLR(optimizer,
                                    start_factor = 0.1,
                                    end_factor = 1.0,
                                    total_iters = warmup_steps)]
        print(f"스케줄러 적용: LinearLR, {warmup_epochs} 에폭")

    if scheduler_config['cosine']['anneal']:
        scheduler_list += [CosineAnnealingLR(optimizer,
                                             T_max = cosine_steps,
                                             eta_min = cosine_eta_min)]
        print(f"스케줄러 적용: Cosine Annealing 주기 {cosine_period} 에폭")

    if scheduler_config['cosine']['restart']:
        scheduler_list += [CosineAnnealingWarmRestarts(optimizer,
                                                       T_0 = cosine_steps,
                                                       eta_min = cosine_eta_min)]
        print(f"스케줄러 적용: Cosine Annealing Warm Restarts 주기 {cosine_period} 에폭")

    if len(scheduler_list) == 1:
        scheduler = scheduler_list[0]
    else:
        scheduler = SequentialLR(
            optimizer,
            schedulers=scheduler_list,
            milestones=[warmup_steps]
        )

    return scheduler