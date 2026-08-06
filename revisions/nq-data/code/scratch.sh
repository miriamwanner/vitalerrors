

http://h205:54101/v1



# Together AI
python code/run_importance_ranking.py --input wrong.jsonl --model-url https://api.together.xyz/v1 --model-name openai/gpt-oss-120b --api-key $TOGETHER_API_KEY --test




openai/gpt-oss-120b








python code/compare_rankings.py \
    --systems \
        gpt-4o:normal.jsonl \
        llama-3.3-70b:model-outputs/meta-llama_Llama-3.3-70B-Instruct-Turbo/normal.jsonl \
        qwen-2.5-72b:model-outputs/Qwen2.5-72B-Instruct/normal.jsonl \
        deepseek-r1:model-outputs/DeepSeek-R1-Distill-Llama-70B/normal.jsonl \
        gpt-oss-120b:model-outputs/gpt-oss-120b/normal.jsonl \
        human:normal_human-annotated.jsonl \
    --output results/normal_agreement.txt \
    [--per-instance]