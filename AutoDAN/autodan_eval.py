import os, sys
import gc
import json
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AutoDAN"))
from utils.eval_utils import _sample_next_suffixes, _prepare_reference, log_init, set_seed, check_for_attack_success
from utils.references import TEST_PREFIXES, MODEL_PATH_DICTS
from utils.opt_utils import (
    get_score_autodan,
    load_model_and_tokenizer,
)
from utils.string_utils import autodan_SuffixManager, load_conversation_template

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_elites", type=float, default=0.05)
    parser.add_argument("--crossover", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--attack_mode", type=str, choices=["ga", "hga"], default="ga")
    ## Note that this argument is only used in the case of hg
    parser.add_argument("--iter", type=int, default=5)

    parser.add_argument("--mutation", type=float, default=0.01)
    parser.add_argument("--init_prompt_path", type=str, default="./assets/autodan_initial_prompt.txt")
    parser.add_argument("--dataset_path", type=str, default="./data/advbench/harmful_behaviors.csv")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--save_suffix", type=str, default="normal")
    parser.add_argument("--API_key", type=str, default=None)
    return parser

def run_autodan_eval(args, attack_mode="ga"):
    if attack_mode not in {"ga", "hga"}:
        raise ValueError("attack_mode must be 'ga' or 'hga'")

    set_seed()
    device = f"cuda:{args.device}"
    model_path = os.path.expanduser(MODEL_PATH_DICTS[args.model])
    print(f"model path is {model_path}")

    template_name = args.model
    batch_size = args.batch_size
    num_steps = args.num_steps
    num_elites = max(1, int(batch_size * args.num_elites))

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
    )
    conv_template = load_conversation_template(template_name)
    harmful_data = pd.read_csv(args.dataset_path)
    dataset = zip(harmful_data.goal[args.start:], harmful_data.target[args.start:])
    infos = {}
    crit = nn.CrossEntropyLoss(reduction="mean")
    reference_template = torch.load("assets/prompt_group.pth", map_location="cpu")
    result_dir = "./results/autodan_ga" if attack_mode == "ga" else "./results/autodan_hga"
    hga_interval = getattr(args, "iter", None)

    for i, (goal, target) in tqdm(enumerate(dataset), total=len(harmful_data.goal[args.start:])):
        reference = _prepare_reference(reference_template, template_name)
        log = log_init()
        info = {
            "goal": goal,
            "target": target,
            "final_suffix": "",
            "final_respond": "",
            "total_time": 0,
            "is_success": False,
            "log": log,
        }

        start_time = time.time()
        new_adv_suffixs = reference[:batch_size]
        word_dict = {}
        adv_suffix = ""
        gen_str = ""
        is_success = False

        for step_index in range(num_steps):
            with torch.no_grad():
                epoch_start_time = time.time()
                losses = get_score_autodan(
                    tokenizer=tokenizer,
                    conv_template=conv_template,
                    instruction=goal,
                    target=target,
                    model=model,
                    device=device,
                    test_controls=new_adv_suffixs,
                    crit=crit,
                )
                score_list = losses.cpu().numpy().tolist()
                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffixs[best_new_adv_suffix_id]
                current_loss = losses[best_new_adv_suffix_id]

                adv_suffix = best_new_adv_suffix
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
                    TEST_PREFIXES,
                )

                new_adv_suffixs, word_dict = _sample_next_suffixes(
                    attack_mode=attack_mode,
                    step_index=step_index,
                    word_dict=word_dict,
                    new_adv_suffixs=new_adv_suffixs,
                    score_list=score_list,
                    num_elites=num_elites,
                    batch_size=batch_size,
                    crossover=args.crossover,
                    num_points=args.num_points,
                    mutation=args.mutation,
                    API_key=args.API_key,
                    reference=reference,
                    hga_interval=hga_interval,
                )

                epoch_cost_time = round(time.time() - epoch_start_time, 2)
                print(
                    "################################\n"
                    f"Current Data: {i}/{len(harmful_data.goal[args.start:])}\n"
                    f"Current Epoch: {step_index}/{num_steps}\n"
                    f"Passed:{is_success}\n"
                    f"Loss:{current_loss.item()}\n"
                    f"Epoch Cost:{epoch_cost_time}\n"
                    f"Current Suffix:\n{best_new_adv_suffix}\n"
                    f"Current Response:\n{gen_str}\n"
                    "################################\n"
                )

                info["log"]["time"].append(epoch_cost_time)
                info["log"]["loss"].append(current_loss.item())
                info["log"]["suffix"].append(best_new_adv_suffix)
                info["log"]["respond"].append(gen_str)
                info["log"]["success"].append(is_success)

                if is_success:
                    break
                gc.collect()
                torch.cuda.empty_cache()

        info["total_time"] = round(time.time() - start_time, 2)
        info["final_suffix"] = adv_suffix
        info["final_respond"] = gen_str
        info["is_success"] = is_success
        infos[i + args.start] = info

        os.makedirs(result_dir, exist_ok=True)
        with open(f"{result_dir}/{args.model}_{args.start}_{args.save_suffix}.json", "w") as json_file:
            json.dump(infos, json_file, ident=4)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_autodan_eval(args, attack_mode=args.attack_mode)
