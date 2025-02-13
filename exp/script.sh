
conda deactivate
tmux
conda activate SafeDecoding_ENV
cd /home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/exp
python exp_script.py
python defense.py --model_name mistral --attacker HEx-PHI --defender Layer --GPT_API sk-proj

python defense.py --model_name llama2 --attacker Just-Eval --defender PPL --GPT_API sk-proj --test

python defense.py --model_name llama2 --attacker Just-Eval --defender PPL --GPT_API sk-proj 

tmux 
CUDA_VISIBLE_DEVICES=5 python defense.py --model_name llama2 --attacker Just-Eval --defense_off  --GPT_API sk-proj

CUDA_VISIBLE_DEVICES=6 python defense.py --model_name mistral --attacker Just-Eval --defense_off  --GPT_API sk-proj

CUDA_VISIBLE_DEVICES=0 python defense.py --model_name mistral --attacker Just-Eval --defender LayerBugFixer --seed 888 --GPT_API sk-proj
CUDA_VISIBLE_DEVICES=4 python defense.py --model_name mistral --attacker AdvBench --defender LayerBugFixer --seed 888 --GPT_API sk-proj


CUDA_VISIBLE_DEVICES=1 python defense.py --only_eval --model_name mistral --attacker MMLU --defense_off --eval_input_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/exp_outputs_new/PPL_llama2_AdvBench_104_2024-10-15 12:35:49/PPL_llama2_AdvBench_104_2024-10-15 12:35:49.json" --GPT_API sk-proj 

CUDA_VISIBLE_DEVICES=1 python defense.py --only_eval --model_name mistral --attacker AdvBench --defender PPL --eval_input_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/exp_outputs_new/PPL_llama2_AdvBench_104_2024-10-15 12:35:49/PPL_llama2_AdvBench_104_2024-10-15 12:35:49.json" --GPT_API sk-proj


CUDA_VISIBLE_DEVICES=1 python MMLU.py --model_name mistral --attacker MMLU --defense_off --mmlu_data_dir "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Kangaroo/data/MMLU_data" --mmlu_save_dir "/home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/mmlu_output" --GPT_API sk-proj
CUDA_VISIBLE_DEVICES=1 python MMLU.py --model_name llama2 --attacker MMLU --defense_off --mmlu_data_dir "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Kangaroo/data/MMLU_data" --mmlu_save_dir "/home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/mmlu_output" --GPT_API sk-proj

CUDA_VISIBLE_DEVICES=1 python MMLU.py --model_name mistral --attacker MMLU --defender LayerBugFixer --mmlu_data_dir "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Kangaroo/data/MMLU_data" --mmlu_save_dir "/home/kz34/Yang_Ouyang_Projects/ICLR2025/jailbreaking_related/SafeDecoding/mmlu_output" --GPT_API sk-proj

CUDA_VISIBLE_DEVICES=2 python defense.py --model_name mistral --attacker MMLU --defender LayerBugFixer --GPT_API sk-proj

python MMLU_process_data.py \
    --save-file attack_prompts_MMLU_0train_llama2.json \
    --data-dir /home/kz34/Yang_Ouyang_Projects/ICLR2025/Kangaroo/data/MMLU_data \
    --selected-categories STEM humanities "social sciences" "other (business, health, misc.)" \
    --ntrain 0


CUDA_VISIBLE_DEVICES=2 python MMLU.py --model_name mistral --attacker MMLU --defense_off --GPT_API sk-proj\ --test

CUDA_VISIBLE_DEVICES=1 python defense.py --model_name mistral --attacker GCG --defender Retokenization+LayerBugFixer --GPT_API sk-proj\

CUDA_VISIBLE_DEVICES=3 python MMLU.py --model_name mistral --attacker MMLU --defender Unlearning --GPT_API sk-proj\

CUDA_VISIBLE_DEVICES=2 python MMLU.py --model_name mistral --attacker MMLU --defender SafeDecoding --GPT_API sk-proj\

CUDA_VISIBLE_DEVICES=2 python MMLU.py --model_name mistral --attacker MMLU --defender Paraphrase --GPT_API sk-proj\

CUDA_VISIBLE_DEVICES=2 python MMLU.py --model_name mistral --attacker MMLU --defender Self-Exam --GPT_API sk-proj\

CUDA_VISIBLE_DEVICES=3 python MMLU.py --model_name llama2 --attacker MMLU --defense_off --GPT_API sk-proj\ --test


CUDA_VISIBLE_DEVICES=1 python MMLU.py --model_name mistral --attacker MMLU --defender Paraphrase --paraphase_model self --GPT_API sk-sk

CUDA_VISIBLE_DEVICES=2 python MMLU.py --model_name llama2 --attacker MMLU --defender Paraphrase --paraphase_model self --GPT_API sk-


CUDA_VISIBLE_DEVICES=2 python cipher_attack.py --model_name llama2 --attacker Leetspeak --defender Paraphrase --paraphase_model self --GPT_API sk-sk

CUDA_VISIBLE_DEVICES=2 python defense_combine.py --model_name mistral --attacker GCG --defender Paraphrase --paraphase_model self --GPT_API sk-s

defense_combine