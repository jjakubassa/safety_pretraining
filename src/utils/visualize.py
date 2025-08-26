import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os
from sklearn.decomposition import PCA
from tqdm import tqdm


def collect_hidden_states(model, tokenizer, prompts, layer_idx=None):
    model.eval()

    layer_hidden_states = {}

    for prompt in tqdm(prompts, desc="Collecting hidden states"):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True
        )

        # Fix tensor types
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_token_idx = inputs["attention_mask"].sum(dim=1).item() - 1

        # Process each layer
        for i, layer_output in enumerate(outputs.hidden_states):
            if i not in layer_hidden_states:
                layer_hidden_states[i] = []

            final_hidden = layer_output[0, last_token_idx, :]
            layer_hidden_states[i].append(final_hidden.cpu())

    return {layer: torch.stack(states) for layer, states in layer_hidden_states.items()}


def get_pca_df(harmful_states, harmless_states, n_components=10):
    """
    Create a DataFrame with PCA results for harmful and harmless states.

    Args:
        harmful_states: Tensor of hidden states for harmful prompts
        harmless_states: Tensor of hidden states for harmless prompts
        n_components: Number of PCA components to compute

    Returns:
        DataFrame with PCA results and category labels
    """
    # Convert tensors to float32 before concatenating
    harmful_states = harmful_states.float()
    harmless_states = harmless_states.float()

    all_states = torch.cat([harmful_states, harmless_states], dim=0)
    labels = ["harmful"] * harmful_states.size(0) + [
        "harmless"
    ] * harmless_states.size(0)

    # Convert to numpy array for sklearn
    all_states_np = all_states.cpu().numpy()

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(all_states_np)

    df = pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(n_components)])
    df['Category'] = labels
    return df


def scatter_first_two_components(df: pd.DataFrame, save_path=None, title=None):
    """
    Create a scatter plot of the first two PCA components.

    Args:
        df: DataFrame with PCA results
        save_path: Path to save the figure (if None, figure is only displayed)
        title: Title for the plot
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=df,
        x='PC1',
        y='PC2',
        hue='Category',
        palette={'harmful': 'red', 'harmless': 'blue'},
        alpha=0.7,
        s=80,
        edgecolor='none'
    )

    plt.title(title or "PCA of final-token hidden states", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    sns.move_legend(ax, "upper right", frameon=True, title="Category")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
