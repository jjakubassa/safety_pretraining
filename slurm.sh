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
model1abl="models/$model1-abliterated"

# Trained on raw data (same as model 3 above)
# mix_ift_v4-smollm2-1.7b-all_raw_folders_baseline-600B
model2="locuslab/mix_ift_v4-smollm2-1.7b-all_raw_folders_baseline-600B"
model2abl="models/$model2-abliterated"

# Score 0 data (safe data) only:
# locuslab/mix_ift_v4-smollm2-1.7b-score0_only-600B
model3="locuslab/mix_ift_v4-smollm2-1.7b-score0_only-600B"
model3abl="models/$model3-abliterated"

# Score 0 + Rephrase data
# mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning-600B
model4="locuslab/mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning-600B"
model4abl="models/$model4-abliterated"

# Score 0 + Rephrase data + Metatags
# mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning_metadata-600B
model5="locuslab/mix_ift_v4-smollm2-1.7b-score0_mix_rephrased_from_beginning_metadata-600B"
model5abl="models/$model5-abliterated"

# Score 0 + Rephrase data + Refusals
# mix_ift_v4-smollm2-1.7b-base-score0_mix_rephrase123_with_mild_refusal45-600B
model6="locuslab/mix_ift_v4-smollm2-1.7b-base-score0_mix_rephrase123_with_mild_refusal45-600B"
model6abl="models/$model6-abliterated"

# Score 0 + Rephrase data + Refusals + Metatags
# mix_ift_v9-smollm2-1.7b-score0_rephrase123_mild_ref45_metadata_5p-600B-metamix3p-1k-0
model7="locuslab/mix_ift_v9-smollm2-1.7b-score0_rephrase123_mild_ref45_metadata_5p-600B-metamix3p-1k-0"
model7abl="models/$model7-abliterated"

model8="zai-org/glm-4-9b-chat-hf"
model8abl="models/$model8-abliterated"

model9="meta-llama/Llama-3.3-70B-Instruct"
model9abl="models/$model9-abliterated"

model10="Qwen/Qwen3-14B"
model10abl="models/$model10-abliterated"

echo $(date)

# for idx in 10 9 8 7 6 5 4 3 2 1; do
#     eval "model=\$model${idx}"
#     eval "modelabl=\$model${idx}abl"
#     cmd="uv run src/abliterate.py -m $model -o $modelabl"
#     echo $cmd
#     $cmd
# done

# evaluate
cmd="uv run src/evaluate.py --model_names $model1 $model1abl $model2 $model2abl $model3 $model3abl $model4 $model4abl $model5 $model5abl $model6 $model6abl $model7 $model7abl $model8 $model8abl $model9 $model9abl $model10 $model10abl"
echo $cmd
$cmd
