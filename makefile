
MODEL_NAME = meta-llama/Llama-2-7b-chat-hf


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
		--smoothllm_num_copies 2 \
		--verbose
