conda activate SafeDecoding_ENV
cd /home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/exp
CUDA_VISIBLE_DEVICES=1 python defense.py --only_eval --model_name mistral --attacker Just-Eval --defense_off --eval_input_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/exp_outputs_new/PPL_llama2_AdvBench_104_2024-10-15 12:35:49" --GPT_API sk-proj 

CUDA_VISIBLE_DEVICES=1 