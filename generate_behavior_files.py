#import jailbreakbench as jbb
import json

model_name = "llama-2-7b-chat-hf"
number_behaviors = 2

if False:
    artifact = jbb.read_artifact(
        method="GCG",
        model_name=model_name
    )


goal,target, control= [],[],[]
examples = 0

autodan_output_file = "AutoDAN/results/autodan_hga/llama2_0_normal.json"

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

if False:
    for i in range(len(artifact.jailbreaks)):
        if artifact.jailbreaks[i].jailbroken:
            print(artifact.jailbreaks[i])
            goal.append(artifact.jailbreaks[i].prompt)
            target.append(artifact.jailbreaks[i].response.split("\n\n")[0])
            control.append('') # No control it's in the goal
            examples += 1
        if examples >= number_behaviors:
            break


behaviors = {
    "goal": goal,
    "target": target,
    "controls": control
}

print(f"Found {examples} examples of jailbreak behavior for {model_name}. Saving to json.")
import json

save_file = f"data/AutoDAN/{model_name}_behaviors.json"

with open(save_file, "w") as f:
    json.dump(behaviors, f, indent=4)
