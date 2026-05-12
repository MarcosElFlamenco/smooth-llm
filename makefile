
MODEL_NAME = lmsys/vicuna-13b-v1.5

download_model:
	hf download $(MODEL_NAME)


TARGET_MODEL = llama2

smooth_llm:
	python main.py \
		--results_dir ./results \
		--target_model $(TARGET_MODEL) \
		--attack GCG \
		--attack_logfile data/GCG/llama2_behaviors.json \
		--smoothllm_pert_type RandomSwapPerturbation \
		--smoothllm_pert_pct 10 \
		--smoothllm_num_copies 10