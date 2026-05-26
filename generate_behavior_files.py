import json



model_name = "llama-2-7b-chat-hf"
attack_mode = "hga"
number_behaviors = 2

goal,target, control= [],[],[]
examples = 0


autodan_output_file = f"AutoDAN/results/autodan_{attack_mode}/llama2_0_normal.json"

with open(autodan_output_file, "r") as f:
    data = json.load(f)

for key in data.keys():
    line = data[key]
    print(f'Example {key}')
    if line['is_success']:
        print("Success!")
        goal.append(line['goal'])
        target.append(line['target'])
        control.append(line['final_suffix'])
        examples += 1


behaviors = {
    "goal": goal,
    "target": target,
    "controls": control
}

print(f"Found {examples} examples of jailbreak behavior for {model_name}. Saving to json.")

save_file = f"data/AutoDAN/{model_name}_{attack_mode}_behaviors.json"

with open(save_file, "w") as f:
    json.dump(behaviors, f, indent=4)
