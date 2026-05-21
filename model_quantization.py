import os

# 1. Save quantized model somewhere temporary
model.save_pretrained("/tmp/my-quantized-model")
tokenizer.save_pretrained("/tmp/my-quantized-model")

# 2. Find where HF would cache a real model ID you'll "hijack"
#    Pick any real but unused model, e.g. "bert-base-uncased" or a dummy you know exists
fake_id = "bert-base-uncased"  # just an example — pick one you won't actually use
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
target = f"{cache_dir}/models--{fake_id.replace('/', '--')}/snapshots/main"

os.makedirs(os.path.dirname(target), exist_ok=True)

# 3. Symlink it
os.symlink("/tmp/my-quantized-model", target)