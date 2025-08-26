#!/usr/bin/env python3
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from tyro import cli

console = Console()

def interactive_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each record in the dataframe, displays the question and response,
    and prompts the user to classify the response using a single key:
      - [bold]r[/bold] for REFUSAL
      - [bold]n[/bold] for NON-REFUSAL
    After selecting a classification, the user is prompted for an optional note.
    The classification is added as a boolean column "is_refusal_user" (True for REFUSAL)
    and the note is stored in column "user_note".
    """
    classifications = []
    notes = []
    total = len(df)

    for idx in range(total):
        row = df.iloc[idx]
        console.rule(f"[bold blue]Example {idx + 1} of {total}")
        console.print(f"[bold]Model:[/bold] {row['model']}")
        console.print(f"[bold]Label:[/bold] {row['label']}")
        console.print(f"[bold]Question:[/bold] {row['question']}")
        console.print(f"[bold]Response:[/bold]\n{row.get('response', '(no response found)')}\n")

        prompt_text = ("Enter classification ([bold red]r[/bold red] = REFUSAL, "
                       "[bold green]n[/bold green] = NON-REFUSAL)")
        while True:
            user_input = Prompt.ask(prompt_text).strip().lower()
            if user_input in ("r", "n"):
                break
            console.print("[red]Invalid input. Please enter 'r' for refusal or 'n' for non-refusal.[/red]")
        is_refusal = (user_input == "r")
        classifications.append(is_refusal)

        note = Prompt.ask("Enter any note regarding this response (optional)", default="")
        notes.append(note)

    df["is_refusal_user"] = classifications
    df["user_note"] = notes
    return df

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors the summary function from evaluate.
    It collects all columns starting with "is_refusal" (which may include
    classifications from regex, LLM judges, etc., as well as the user classification)
    and melts the dataframe to aggregate counts per model, label, and judge.
    """
    # Identify all columns that start with "is_refusal"
    refusal_cols = [col for col in df.columns if col.startswith("is_refusal")]
    if not refusal_cols:
        console.print("[red]No refusal classification columns found.[/red]")
        return pd.DataFrame()
    df_long = df.melt(
        id_vars=["model", "label"],
        value_vars=refusal_cols,
        var_name="refusal_judge",
        value_name="is_refusal"
    )
    # Remove the prefix to have short judge names
    df_long["refusal_judge"] = df_long["refusal_judge"].str.replace("^is_refusal_", "", regex=True)
    summary = df_long.groupby(["model", "label", "refusal_judge"])["is_refusal"].agg(
        refused=lambda s: (s == True).sum(),
        not_refused=lambda s: (s == False).sum(),
        total="count"
    ).reset_index()
    return summary

def run(evaluation_csv: str, limit: int = -1, filter_label: str = "all") -> None:
    """
    Loads the evaluation CSV, applies a filter based on the provided filter_label,
    sorts by question to group responses from different models together, and then limits the
    number of distinct questions to process. For each response, the user is prompted for a classification
    (with an optional note). Finally, an aggregated summary is constructed using the same method as in evaluate.

    Args:
        evaluation_csv: Path to the evaluation CSV file.
        limit: Limit on the number of distinct questions (prompts) to evaluate (-1 processes all).
        filter_label: Filter for questions: "harmful", "harmless", or "all".
    """
    try:
        df = pd.read_csv(evaluation_csv)
    except Exception as e:
        console.print(f"[red]Error loading CSV: {e}[/red]")
        return

    required_cols = ["question", "response", "model", "label"]
    for col in required_cols:
        if col not in df.columns:
            console.print(f"[red]Missing required column '{col}' in CSV file.[/red]")
            return

    filter_label = filter_label.lower()
    if filter_label in {"harmful", "harmless"}:
        df = df[df["label"].str.lower() == filter_label]
        if df.empty:
            console.print(f"[yellow]No {filter_label} questions found in the CSV.[/yellow]")
            return
        console.print(f"[bold yellow]Filtering for {filter_label} questions.[/bold yellow]")
    elif filter_label != "all":
        console.print("[red]Invalid filter_label. Must be one of: harmful, harmless, all.[/red]")
        return

    # Sort by question so responses of the same prompt appear together.
    df = df.sort_values("question").reset_index(drop=True)

    # Limit to first N distinct questions, if requested.
    if limit > 0:
        unique_questions = df["question"].drop_duplicates()
        selected_questions = unique_questions.iloc[:limit].tolist()
        df = df[df["question"].isin(selected_questions)]
        console.print(f"[bold yellow]Processing only the first {limit} distinct questions.[/bold yellow]")

    df = interactive_classification(df)
    summary = calculate_metrics(df)

    # Render the summary using a Rich Table.
    table = Table(title="User Classification Summary")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Label", style="magenta")
    table.add_column("Judge", style="yellow")
    table.add_column("Refused", justify="right", style="red")
    table.add_column("Not Refused", justify="right", style="green")
    table.add_column("Total", justify="right", style="bold")

    if not summary.empty:
        for _, row in summary.iterrows():
            table.add_row(
                str(row["model"]),
                str(row["label"]),
                str(row["refusal_judge"]),
                str(row["refused"]),
                str(row["not_refused"]),
                str(row["total"])
            )
        console.print(table)
    else:
        console.print("[red]No summary metrics available.[/red]")

    updated_csv = evaluation_csv.replace(".csv", "_with_user.csv")
    summary_csv = evaluation_csv.replace(".csv", "_user_summary.csv")
    df.to_csv(updated_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    console.print(f"[green]Saved updated evaluation results to {updated_csv}[/green]")
    console.print(f"[green]Saved user classification summary to {summary_csv}[/green]")

if __name__ == "__main__":
    cli(run)
