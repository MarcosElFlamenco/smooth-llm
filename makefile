
MODEL_NAME = meta-llama/Llama-2-7b-chat-hf


download_model:
	hf download $(MODEL_NAME)


TARGET_MODEL = llama2
LOG_FILE = data/AutoDAN/llama-2-7b-chat-hf_behaviors.json


vanilla_inference:
	python main.py \
		--defense_type Empty \
		--results_dir ./results \
		--target_model $(TARGET_MODEL) \
		--attack GCG \
		--attack_logfile $(LOG_FILE) \
		--verbose

smooth_llm:
	python main.py \
		--defense_type SmoothLLM \
		--results_dir ./results \
		--target_model $(TARGET_MODEL) \
		--attack GCG \
		--quantize \
		--attack_logfile $(LOG_FILE) \
		--smoothllm_pert_type RandomSwapPerturbation \
		--smoothllm_pert_pct 10 \
		--smoothllm_num_copies 3 \
		--verbose

ATTACK = AUTODAN

evaluate:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--attack_logfile "AutoDAN/results/autodan_hga/llama2_0_normal.json" \
		--max_new_len 128


autodan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga

generate_behavior_files:
	python generate_behavior_files.py


dan_vanilla_inference:
	python AutoDAN/vanilla_inference.py \
		--input-path data/AutoDAN/llama-2-7b-chat-hf_behaviors.json
		
