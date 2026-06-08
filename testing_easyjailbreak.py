## This is sample code to test the package easyjailbreak
## It didn't work comfortably, i'm giving up this idea for now
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

# First, prepare models and datasets.
path = "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"
attack_model = from_pretrained(model_name_or_path=path,
                               model_name='llama2')
target_model = from_pretrained(model_name_or_path=path,  
                               model_name='llama2')
eval_model = from_pretrained(model_name_or_path=path,
                               model_name='llama2')

dataset = JailbreakDataset('AdvBench')

# Then instantiate the recipe.
attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)

# Finally, start jailbreaking.
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
