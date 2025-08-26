#!/bin/bash
#SBATCH -p gpu24
#SBATCH --gres gpu:1
#SBATCH -o /BS/safety_pretraining/work/abliteration/logs/slurm-%A_%a.out
#SBATCH -t 0-4:00
#SBATCH --job-name=abliteration
#SBATCH --mail-user=jjakubas@mpi-inf.mpg.de
#SBATCH --mail-type=END,FAIL

# setup
# referecence model
model1="HuggingFaceTB/SmolLM2-1.7B-Instruct"
model1abl="models/HuggingFaceTB/SmolLM2-1.7B-Instruct-abliterated"

# only rephrasing no refusal
model2="locuslab/mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning-600B"
model2abl="models/locuslab/mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning-600B-abliterated"

# one of the best models
model3="locuslab/mix_ift_v9-smollm2-1.7b-score0_rephrase123_mild_ref45_metadata_5p-600B-metamix3p-1k-0"
model3abl="models/locuslab/mix_ift_v9-smollm2-1.7b-score0_rephrase123_mild_ref45_metadata_5p-600B-metamix3p-1k-0-abliterated"

# model4="zai-org/glm-4-9b-chat-hf"
# model4abl="zai-org/glm-4-9b-chat-hf-abliterated"

# second reference
model5="Qwen/Qwen3-14B"
model5abl="models/Qwen/Qwen3-14B-abliterated"

abl_args="--data-harmful data/harmful_train.parquet --data-harmless data/harmless_train.parquet --refusal_dir_fun=MEAN"

echo $(date)

# abliterate
# cmd="uv run src/abliterate.py --model-name $model1 --output $model1abl $abl_args"
# echo $cmd
# $cmd

# cmd="uv run src/abliterate.py --model-name $model2 --output $model2abl $abl_args"
# echo $cmd
# $cmd

# cmd="uv run src/abliterate.py --model-name $model3 --output $model3abl $abl_args"
# echo $cmd
# $cmd

# # cmd="uv run src/abliterate.py --model-name $model4 --output $model4abl $abl_args"
# # echo $cmd
# # $cmd

# cmd="uv run src/abliterate.py --model-name $model5 --output $model5abl $abl_args"
# echo $cmd
# $cmd

# cmd="uv run src/old/abliterate.py -m $model1 -o $model1abl"
# echo $cmd
# $cmd

# cmd="uv run src/old/abliterate.py -m $model2 -o $model2abl"
# echo $cmd
# $cmd
# cmd="uv run src/old/abliterate.py -m $model3 -o $model3abl"
# echo $cmd
# $cmd

# # cmd="uv run src/old/abliterate.py -m $model4 -o $model4abl"
# # echo $cmd
# # $cmd

# cmd="uv run src/old/abliterate.py -m $model5 -o $model5abl"
# echo $cmd
# $cmd

# evaluate
cmd="uv run src/evaluate.py --model_names $model1 $model1abl $model2 $model2abl $model3 $model3abl $model5 $model5abl" # $model4 $model4abl
$cmd
