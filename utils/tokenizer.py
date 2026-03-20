from transformers import AutoTokenizer, AddedToken

def build_custom_tokenizer(config):

    model_name      = config['model']['name']
    new_tokens      = config['tokenizer']['new_tokens']
    special_tokens  = config['tokenizer']['special_tokens']

    tok = AutoTokenizer.from_pretrained(model_name)

    if new_tokens is not None:
        new_tokens = [t for t in new_tokens if t not in tok.get_vocab()]
        print(f"new tokens: {new_tokens}")

        new_tokens = [AddedToken(t, single_word=False, lstrip=False, rstrip=False, normalized=False) for t in new_tokens]
        tok.add_tokens(new_tokens)
        
    if special_tokens is not None:
        tok.add_special_tokens({"additional_special_tokens": special_tokens})

    return tok