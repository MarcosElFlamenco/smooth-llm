import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AutoDAN"))
import torch
import pandas as pd
import json
import random
import time
from tqdm.auto import tqdm
import argparse
import time
#import lib.perturbations as perturbations
#import lib.attacks as attacks
#import lib.language_models as language_models
#import lib.model_configs as model_configs
from defenses.defense_factory import get_defense
from attacks.attack_factory import get_attack
import numpy as np
import torch.nn as nn

from utils.opt_utils import load_model_and_tokenizer
from utils.string_utils import load_conversation_template
from utils.eval_utils import check_for_attack_success, set_seed
from utils.references import MODEL_PATH_DICTS
from jailbreak_evaluators import SyntaxicEvaluator

def get_attack_data(attack_data_path):
    with open(attack_data_path, "r") as f:
        artifact_dataset = json.load(f)
    return artifact_dataset


def main(args):
    start_time = time.time()

    # Preparing output dictionary
    os.makedirs(args.results_dir, exist_ok=True)
    results = {}

    # Setup compute
    set_seed()
    device = f"cuda:{args.device}"

    # Setup conv template
    template_name = args.target_model
    conv_template = load_conversation_template(template_name)

    # Setup model
    model_path = os.path.expanduser(MODEL_PATH_DICTS[args.target_model])
    target_model , tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
    )
    print(f"Success: loaded model from path {model_path}")

    # Setup defense
    defense = get_defense(
        defense_type=args.defense_type,
        target_model=target_model,
        tokenizer=tokenizer,
        conv_template=conv_template,
        args=args
    )

    # Setup jailbreak evaluator
    jailbreak_evaluator = SyntaxicEvaluator()
    print(f"Using jailbreak evaluator: {jailbreak_evaluator.__class__.__name__}")

    # Setup attack dataset
    
    attack = get_attack(
        args.attack, 
        logfile = args.attack_logfile, 
        target_model = target_model, 
        tokenizer = tokenizer, 
        conv_template = conv_template
    )

    num_jailbroken = 0

    start_time = time.time()
    artifact_start_time = start_time

    for i, (input_ids, assistant_role_slice) in enumerate(attack.prompts):
        print(f"Evaluating artifact {i}...")
        output = defense(input_ids = input_ids,
                         assistant_role_slice=assistant_role_slice,
                         gen_config=None,
                         batch_size=64, 
                         max_new_len=args.max_new_len,
                        )
        
        jailbroken = jailbreak_evaluator(output)
        if jailbroken:
            num_jailbroken += 1
        artifact_inference_time = time.time() - artifact_start_time
        artifact_start_time = time.time()
        print(f"######################## OUTPUT ########################: \n {output} \n\n  ######################## JAILBROKEN: {jailbroken} \n INFERENCE TIME: {artifact_inference_time}s")

        result = {
            "goal": artifact["goal"],
            "target": artifact["target"],
            "adv_suffix": artifact["final_suffix"],
            "output": output,
            "time": artifact_inference_time,
            "jailbroken": jailbroken
        }

        results[str(i)] = result

        with open(f"{args.result_dir}/{args.model}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

    print(f"Total inference time: {time.time() - start_time} seconds")
    print(f"Number of jailbroken artifacts: {num_jailbroken} out of {len(attack_data)}")


 
if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./defense_testing'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='llama2',
        choices=['vicuna', 'llama2']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='AUTODAN',
        choices=['AUTODAN', 'PAIR', 'GCG']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='data/GCG/vicuna_behaviors.json'
    )

    parser.add_argument(
        "--num_artifacts_to_eval",
        type=int,
        default=100000
    )
    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--verbose',
        action="store_true"
    )
    parser.add_argument(
        '--quantize',
        action="store_true"
    )

    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
    )

    parser.add_argument(
        '--defense_type',
        type=str,
        default='NoDefense',
        choices=[
            'NoDefense',
            'SmoothLLM'
        ]
    )

    parser.add_argument(
        "--device", 
        type=int, 
        default=0
    )

    parser.add_argument(
        "--max_new_len", 
        type=int, 
        default=64
    )


    args = parser.parse_args()
    main(args)
