
# Overview

To run everything (using slurm cluster):

```bash
sbatch slurm.sh
```

This includes:
- abliteration of models specified in slurm.sh with a "training set" of prompts
    - model weights saved to `models` folder
- generate responses for each model with and without abliteration
- evaluation of responses. classfication of response into refusal/no refusal based on a) regex and b) a llm judge. a simple analysis of classification results is done (e.g. confusion matrix)
    - logs with all responses and confusion matrix can be found at `logs` folder
    - csv with responses and classification results can be found at `results` folder
    - csv with summary statistics can be found at `results` folder

Hints:
- it might be nessecary to login to the hugginface account and accept conditions for the use of some llms (e.g. the default "mistralai/Ministral-8B-Instruct-2410")
- requirements.txt might be not uptodate

Seperation of data used for model abliteration and for evaluation is done with `split_data.py` script.

Everything below is copied from the original repo and gives more details about the abliteration script.

# Abliteration

Make abliterated models using transformers, easy and fast.

## Introduction

There exist some directions that make LLMs to refuse users' input. Abliteration is a technique that can calculate the most significant refusal directions with harmful and harmless prompts, and then remove them from the model. This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens.

The code has been tested on Llama-3.2, Qwen2.5-Coder, Ministral-8b.

VRAM/RAM requirements: This repository has been making efforts to reduce VRAM usage. You can abliterate whatever model you want, as long as it fits in your VRAM. Loading model in 4-bit precision using bitsandbytes is recommended for large models if you have limited VRAM. However, I always assume that you have enough memory to load the **bf16** model.

> [!NOTE]
> Abliteration is not uncensorment. Though abliterated, it doesn't necessarily mean the model is completely uncensored, it simply will not explicitly refuse you, theoretically.

## Quick Start

### Clone the repositoty

```shell
git clone https://github.com/Orion-zhen/abliteration.git && cd abliteration
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Make your abliterations

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir>
```

### Chat with your abliterated model

```shell
python chat.py -m <path_to_your_abliterated_model>
```

### Compare between models

```shell
python compare.py -a <model_a> -b <model_b>
```

### Examples

- Abliterate Llama-3.2:

```shell
python abliterate.py -m meta-llama/Llama-3.2-3B-Instruct -o llama3.2-3b-abliterated
```

- Load model in 4-bit precision using bitsandbytes:

```shell
python abliterate.py -m meta-llama/Llama-3.2-3B-Instruct -o llama3.2-3b-abliterated --load-in-4bit
```

- Compare your abliterated model with the original model:

```shell
python compare.py -a meta-llama/Llama-3.2-3B-Instruct -b llama3.2-3b-abliterated
```

- Compare in 4-bit precision using bitsandbytes:

```shell
python compare.py -a meta-llama/Llama-3.2-3B-Instruct -b llama3.2-3b-abliterated --load-in-4bit
```

> [!NOTE]
> If you use `--load-in-4bit` or `--load-in-8bit`, then I will assume you are lack of VRAM, and the final appliance step will be performed with CPU and memory. Please make sure you have enough memory to load the **bf16** model.

Now your model will be abliterated and saved to `<output_dir>`. Once it finishes, you can immediately chat with your abliterated model in the terminal. For Chinese models, you can use `--deccp` to abliterate it from certain topics.

## Advanced Usage

### Use config files

This repository now supports `.json` config file. This file should contain a `dict` of config key value pairs. For example:

```json
{
    "model": "/absolute/path/to/your/model",
    "output": "/output/dir",
    "data-harmful": "/absolute/path/to/harmful-prompts.txt",
    "scale-factor": 114,
    "load-in-4bit": true
}
```

```shell
python abliterate.py -c config.json
```

Loading config file will **overwrite** command line arguments.

### Use your own prompts

You can use your own prompts to abliterate your model. Supported file formats are `.txt`, `.parquet` and `.json`. Detailed formats are listed below:

- `.txt`: Each line of the file is a prompt
- `.parquet`: A parquet file with column `text`
- `.json`: A json file with list of strings

Then load your own prompts using `--data-harmful` and `--data-harmless` arguments:

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --data-harmful /path/to/my/harmful.txt --data-harmless /path/to/my/harmless.txt
```

### Scale factor

You can use `--scale-factor` to control the abliteration strength. A scale factor larger then 1 will impose stronger removal of refusals, while a negative scale factor will encourage refusal. You can try to increase the scale factor to see if it helps.

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --scale-factor 1.5
```

### Input/Output refusals

You can output the refusals to a file using `--output-refusals` argument:

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --output-refusals refusals.bin
```

And load the refusals back using `--load-refusals` argument:

```shell
python abliterate.py -m <path_to_your_model> --input-refusals refusals.bin -o <output_dir>
```

If `--input-refusal` is provided, the script will not compute refusal directions again.

### Abliterate specific targets

By default, abliteration will be applied to `o_proj` and `down_proj`. You can add more targets by modifying the code below, as long as it won't mess up the model:

```python
# utils/apply.py, apply_abliteration()
lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
  lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
  refusal_dir,
  scale_factor,
)
lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
  lm_model.layers[layer_idx].mlp.down_proj.weight.data,
  refusal_dir,
  scale_factor,
)
```

Available targets can be found in [transformers model architectures](https://github.com/huggingface/transformers/tree/main/src/transformers/models) and [mergekit model architectures](https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures).

### Best practices

This repository provides a bunch of parameters to optimize. To get the best results, you can try the following steps:

1. Carefully choose your prompts. Prompts in this repository is a general example, you can use your own prompts to get better results.
2. Adjust parameters. The script provides various parameters to control the abliteration progress. You can try different values to see if it helps.
3. Change the targets. You can modify the code to abliterate other targets, as long as it won't mess up the model.
4. If you have limited VRAM, try `--load-in-4bit` or `--load-in-8bit` to load the model in 4-bit or 8-bit precision.

### Full arguments

Use `--help` to see all available arguments:

```shell
python abliterate.py --help
```

## Credits

- [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)
- [huihui-ai](https://huggingface.co/huihui-ai)
