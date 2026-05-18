##This model is for testing code
MODELS = {
    "llama2": {
        'model_path': 'meta-llama/Llama-3.2-1B',
        'tokenizer_path': 'meta-llama/Llama-3.2-1B',
        'conversation_template': 'llama-3'  ## This probably won't exist, I'll wait for the warning
    }
}
LARGE_MODELS = {
    'llama2': {
        'model_path': 'meta-llama/Llama-2-7b-chat-hf',
        'tokenizer_path': 'meta-llama/Llama-2-7b-chat-hf',
        'conversation_template': 'llama-2'
    },
    'vicuna': {
        'model_path': 'lmsys/vicuna-13b-v1.5',
        'tokenizer_path': 'lmsys/vicuna-13b-v1.5',
        'conversation_template': 'vicuna'
    }
}
