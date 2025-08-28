#!/usr/bin/env python3
import pandas as pd
from rich.console import Console
from tyro import cli
from interactive_evaluation import calculate_metrics
from openai import OpenAI

def judge_refusal_with_openai(df: pd.DataFrame, logger, batch_size: int = 2) -> pd.DataFrame:
    client = OpenAI()

    results = []
    questions = df["question"].tolist()
    responses = df["response"].tolist()
    total_batches = (len(questions) // batch_size) + (1 if len(questions) % batch_size > 0 else 0)

    # Prompt template is same as in evaluate.py
    prompt_template = {
        "id": "ADD YOUR PROMPT TEMPLATE ID",
        "version": "1",
        "variables": {}
    }

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]
        batch_responses = responses[i : i + batch_size]
        for question, response in zip(batch_questions, batch_responses):
            # Prepare the prompt by updating the template variables.
            cur_prompt = prompt_template.copy()
            cur_prompt["variables"] = {"question": question, "response": response}
            # Allow exceptions to propagate instead of catching them.
            api_response = client.responses.create(prompt=cur_prompt)
            classification = api_response.output_text.strip()
            is_refusal = not "NON-REFUSAL" in classification.upper()
            results.append(is_refusal)
            logger.log(f"### Question:\n {question}\n### Response:\n {response} \nClassification: {classification} -> is_refusal = {is_refusal}\n")
        logger.log(f"Processed batch {(i // batch_size) + 1}/{total_batches} with OpenAI judge")
    df["is_refusal_openai"] = results
    return df

def run(comparison_csv: str) -> None:
    console = Console()
    try:
        df = pd.read_csv(comparison_csv).head()
    except Exception as e:
        console.print(f"[red]Error loading CSV '{comparison_csv}': {e}[/red]")
        raise

    console.print(f"[green]Loaded comparison dataframe with {len(df)} records.[/green]")

    df = judge_refusal_with_openai(df, logger=console)
    summary_df = calculate_metrics(df)
    updated_csv = comparison_csv.replace(".csv", "_with_openai.csv")
    summary_csv = comparison_csv.replace(".csv", "_openai_summary.csv")

    try:
        df.to_csv(updated_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
    except Exception as e:
        console.print(f"[red]Error saving CSV files: {e}[/red]")
        raise

    console.print(f"[green]Updated comparison CSV saved as {updated_csv}[/green]")
    console.print(f"[green]Summary CSV saved as {summary_csv}[/green]")

if __name__ == "__main__":
    cli(run)
