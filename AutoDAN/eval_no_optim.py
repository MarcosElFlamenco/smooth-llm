import argparse
import copy
import gc
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.opt_utils import (
    autodan_sample_control,
    autodan_sample_control_hga,
    get_score_autodan,
    load_model_and_tokenizer,
)
from utils.string_utils import autodan_SuffixManager, load_conversation_template
from utils.eval_utils import check_for_attack_success

SEED = 20

MODEL_PATH_DICTS = {
    "llama2": "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
    "llama3": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "vicuna": "./models/vicuna/vicuna-7b-v1.3",
    "guanaco": "./models/guanaco/guanaco-7B-HF",
    "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
    "mpt-chat": "./models/mpt/mpt-7b-chat",
    "mpt-instruct": "./models/mpt/mpt-7b-instruct",
    "falcon": "./models/falcon/falcon-7b-instruct",
}

DEVELOPER_DICT = {
    "llama2": "Meta",
    "vicuna": "LMSYS",
    "guanaco": "TheBlokeAI",
    "WizardLM": "WizardLM",
    "mpt-chat": "MosaicML",
    "mpt-instruct": "MosaicML",
    "falcon": "TII",
}


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser(include_iter=False):
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_elites", type=float, default=0.05)
    parser.add_argument("--crossover", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=5)
    if include_iter:
        parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--mutation", type=float, default=0.01)
    parser.add_argument("--init_prompt_path", type=str, default="./assets/autodan_initial_prompt.txt")
    parser.add_argument("--dataset_path", type=str, default="./data/advbench/harmful_behaviors.csv")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--save_suffix", type=str, default="normal")
    parser.add_argument("--API_key", type=str, default=None)
    return parser


def get_developer(model_name):
    return DEVELOPER_DICT[model_name]


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(
        input_ids,
        attention_mask=attn_masks,
        generation_config=gen_config,
        pad_token_id=tokenizer.pad_token_id,
        top_p=0.9,
        do_sample=True,
        temperature=0.7,
    )[0]
    return output_ids[assistant_role_slice.stop:]



def log_init():
    return {"loss": [], "suffix": [], "time": [], "respond": [], "success": []}


def run_eval(args, attack_mode="none"):
    if attack_mode not in {"ga", "hga","none"}:
        raise ValueError("attack_mode must be 'ga' or 'hga' or 'none'")

    set_seed()
    device = f"cuda:{args.device}"
    model_path = os.path.expanduser(MODEL_PATH_DICTS[args.model])
    print(f"model path is {model_path}")

    template_name = args.model

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
    )
    conv_template = load_conversation_template(template_name)

    ##artifact load
    path = "results/autodan_ga/llama2_0_normal.json" if attack_mode == "ga" else "results/autodan_hga/llama2_0_normal.json"
    with open(path, "r") as f:
        artifact_dataset = json.load(f)
    example = artifact_dataset["0"]
    goal = example["goal"]
    target = example["target"]
    adv_suffix = example["final_suffix"]


    start_time = time.time()

    with torch.no_grad():

        suffix_manager = autodan_SuffixManager(
            tokenizer=tokenizer,
            conv_template=conv_template,
            instruction=goal,
            target=target,
            adv_string=adv_suffix,
        )

        is_success, gen_str = check_for_attack_success(
            model,
            tokenizer,
            suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
            suffix_manager._assistant_role_slice,
        )

        epoch_cost_time = round(time.time() - start_time, 2)
        print(
            "################################\n"
            f"Current Goal: {goal}\n"
            f"Passed:{is_success}\n"
            f"Current Suffix:\n{adv_suffix}\n"
            f"Current Response:\n{gen_str}\n"
            "################################\n"
        )

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = build_arg_parser(include_iter=True).parse_args()
    run_eval(args, attack_mode="none")
