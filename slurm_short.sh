#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/safety_pretraining/work/abliteration/logs/slurm-%A_%a.out
#SBATCH -t 0-4:00
#SBATCH --job-name=abliteration
#SBATCH --mail-user=jjakubas@mpi-inf.mpg.de
#SBATCH --mail-type=END,FAIL

# setup
# model1="HuggingFaceTB/SmolLM2-1.7B-Instruct"
# model1abl="models/HuggingFaceTB/SmolLM2-1.7B-Instruct-abliterated"

# https://huggingface.co/locuslab/mix_ift_v4-smollm2-1.7b-all_raw_folders_baseline-600B
# model1="locuslab/mix_ift_v4-smollm2-1.7b-all_raw_folders_baseline-600B"
# model1abl="models/locuslab/mix_ift_v4-smollm2-1.7b-all_raw_folders_baseline-600B"

model1="meta-llama/Llama-3.3-70B-Instruct"
model1abl="models/meta-llama/Llama-3.3-70B-Instruct-abliterated"

# model1="zai-org/glm-4-9b-chat-hf"
# model1abl="zai-org/glm-4-9b-chat-hf-abliterated"

# model1="unsloth/gpt-oss-20b-BF16"
# model1abl="unsloth/gpt-oss-20b-BF16-abliterated"

# model1="Qwen/Qwen3-14B"
# model1abl="models/Qwen/Qwen3-14B-abliterated"

# model1="openai/gpt-oss-120b"
# model1abl="openai/gpt-oss-120b-abliterated"

# model1="openai/gpt-oss-20b"
# model1abl="openai/gpt-oss-20b-abliterated"

# abl_args="--data-harmful data/harmful_train.parquet --data-harmless data/harmless_train.parquet --refusal_dir_fun=MEAN --scale_factor=10.0"

# cmd="uv run src/abliterate.py --model-name $model1 --output $model1abl $abl_args"
# echo $cmd
# $cmd

# # evaluate
# cmd="uv run src/evaluate.py --model_names $model1 $model1abl"
# $cmd

## old variant
cmd="uv run src/old/abliterate.py -m $model1 -o $model1abl"
echo $cmd
$cmd

# evaluate
# cmd="uv run src/evaluate.py --model_names $model1 $model1abl"
cmd="uv run src/evaluate.py --model_names $model1 --judge_models=openai/gpt-oss-20b"
echo $cmd
$cmd
