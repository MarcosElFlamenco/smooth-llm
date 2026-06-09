import torch
import numpy as np
import json
import gc
from AutoDAN.utils.string_utils import autodan_SuffixManager
from AutoDAN.utils.eval_utils import generate
#import lib.perturbations as perturbations

## Not used yet, will certainly be used to wrap returns later
class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens

    def perturb(self, perturbation_fn):
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt,
            perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt

class DANPrompt():
    def __init__(self, goal, target, final_suffix, prompt, input_ids, assistant_role_slice):
        self.goal = goal
        self.target = target
        self.final_suffix = final_suffix
        self.text_prompt = prompt
        self.input_ids = input_ids
        self.assistant_role_slice = assistant_role_slice

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model

class AutoDAN(Attack):
    """
    AutoDAN attack
    
    """

    def __init__(self, logfile=None, target_model=None, tokenizer = None, conv_template = None):

        super(AutoDAN, self).__init__(logfile, target_model)

        self.tokenizer = tokenizer
        self.conv_template = conv_template

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        # Enables obj[i]
        self.prompts = [
            self.create_prompt(goal=artifact["goal"], target=artifact["target"], final_suffix=artifact["final_suffix"])
            for artifact in log.values()
        ]

    def create_prompt(self, goal, target, final_suffix):

        suffix_manager = autodan_SuffixManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=goal,
            target=target,
            adv_string=final_suffix,
        )

        prompt = suffix_manager.get_prompt(adv_string=final_suffix)
        input_ids = suffix_manager.get_input_ids_from_prompt(text_prompt=prompt).to(self.target_model.device)
        assistant_role_slice = suffix_manager._assistant_role_slice

        return DANPrompt(goal, target, final_suffix, prompt, input_ids, assistant_role_slice)

### This would have been a nice implementation,
### but the smooth llm code words it differently
"""
    def __getitem__(self, index, tokenizer, conv_template, goal, target, adv_string):
        suffix_manager = autodan_SuffixManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=goal,
            target=target,
            adv_string=adv_suffix,
        )

        artifact = self.data[index]
        adv_string = artifact['adv_string']
        prompt = suffix_manager.get_prompt(adv_string=adv_string)

        return self.data[index]

    # Enables len(obj)
    def __len__(self):
        return len(self.data)
"""
