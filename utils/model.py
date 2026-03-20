from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

def get_model(model_config, tokenizer):
    use_LoRA = model_config['use_LoRA']

    if use_LoRA:
        model_name = model_config['name']
        num_classes = model_config['num_classes']
        classifier_dropout = model_config['classifier_dropout']
        
        r = model_config['r']
        lora_alpha = model_config['lora_alpha']
        target_modules = model_config['target_modules']
        lora_dropout = model_config['lora_dropout']
        bias = model_config['bias']

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            classifier_dropout = classifier_dropout,
        )
        model.resize_token_embeddings(len(tokenizer))

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(f"LoRA + 모델 적용: {model_name}, num_classes={num_classes}, dropout=({classifier_dropout}, {lora_dropout})")

    else:
        model_name = model_config['name']
        num_classes = model_config['num_classes']
        hidden_dropout = model_config['hidden_dropout']
        atten_dropout = model_config['atten_dropout']
        classifier_dropout = model_config['classifier_dropout']

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob = hidden_dropout,
            attention_probs_dropout_prob = atten_dropout,
            classifier_dropout = classifier_dropout,
        )
        model.resize_token_embeddings(len(tokenizer))

        print(f"모델 적용: {model_name}, num_classes={num_classes}, dropout=({hidden_dropout}, {atten_dropout}, {classifier_dropout})")
    return model