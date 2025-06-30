#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/safety_pretraining/work/abliteration/logs/slurm-%A_%a.out
#SBATCH -t 0-4:00
#SBATCH --job-name=abliteration
#SBATCH --mail-user=jjakubas@mpi-inf.mpg.de
#SBATCH --mail-type=END,FAIL

# setup
model1="HuggingFaceTB/SmolLM2-1.7B-Instruct"
model1abl="HuggingFaceTB/SmolLM2-1.7B-Instruct-abliterated"
model2="locuslab/mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning-600B"
model2abl="models/locuslab/mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning-600B-abliterated"
model3="locuslab/mix_ift_v9-smollm2-1.7b-score0_rephrase123_mild_ref45_metadata_5p-600B-metamix3p-1k-0"
model3abl="locuslab/mix_ift_v9-smollm2-1.7b-score0_rephrase123_mild_ref45_metadata_5p-600B-metamix3p-1k-0-abliterated"
abl_args="--scale-factor 1.0 --data-harmful data/harmful_train.parquet --data-harmless data/harmless_train.parquet"

echo $(date)

# abliterate
cmd="uv run abliterate.py -m $model1 -o $model1abl $abl_args"
echo $cmd
$cmd

cmd="uv run abliterate.py -m $model2 -o $model2abl $abl_args"
echo $cmd
$cmd

cmd="uv run abliterate.py -m $model3 -o $model3abl $abl_args"
echo $cmd
$cmd

# evaluate
cmd="uv run evaluate.py --model_names $model1 $model1abl $model2 $model2abl $model3 $model3abl"
$cmd
