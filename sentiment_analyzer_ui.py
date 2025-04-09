#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.box import ROUNDED
from rich.style import Style
from rich import print as rprint

# Import the model functions
from train_eval import run_experiment
from data import generate_sample_data

console = Console()


def print_header():
    title = Text("Sentiment Analysis from Text and Context", style="bold blue")
    subtitle = Text("Using LSTM and CNN-LSTM Models", style="italic cyan")

    console.print(Panel.fit(
        f"{title}\n{subtitle}",
        border_style="blue",
        padding=(1, 10),
        box=ROUNDED
    ))
    console.print()


def get_experiment_config():
    """Interactive prompt to configure the experiment."""
    config = {}

    # Model selection
    model_type = questionary.select(
        "Select model architecture:",
        choices=[
            {"name": "LSTM only (simpler option)", "value": "lstm"},
            {"name": "CNN + LSTM (full requirement)", "value": "cnn_lstm"}
        ]
    ).ask()

    # Embedding selection
    use_pretrained = questionary.confirm(
        "Use pretrained word embeddings (GloVe)?",
        default=False
    ).ask()

    # Attention mechanism
    use_attention = questionary.confirm(
        "Use attention mechanism?",
        default=False
    ).ask()

    # Dataset options
    data_path = questionary.text(
        "Path to dataset CSV (leave empty to generate sample data):",
        default="sentiment_data.csv"
    ).ask()

    # Batch size
    batch_size = questionary.select(
        "Select batch size:",
        choices=["16", "32", "64", "128"],
        default="32"
    ).ask()

    # Epochs
    epochs = questionary.text(
        "Number of training epochs:",
        default="10"
    ).ask()

    # Learning rate
    learning_rate = questionary.select(
        "Select learning rate:",
        choices=["0.0001", "0.001", "0.01"],
        default="0.001"
    ).ask()

    # Set configuration
    config = {
        'data_path': data_path,
        'embedding_path': 'glove.6B.100d.txt',
        'name': f"{model_type.upper()}_{'Pretrained' if use_pretrained else 'Scratch'}_{'Attn' if use_attention else 'NoAttn'}",
        'save_path': f'models/{model_type.upper()}_{"Pretrained" if use_pretrained else "Scratch"}_{"Attn" if use_attention else "NoAttn"}.pt',
        'vocab_size': 5000,
        'embedding_dim': 100,
        'hidden_dim': 128,
        'output_dim': 3,  # Positive, Negative, Neutral
        'filter_sizes': [2, 3, 4],
        'num_filters': 64,
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'weight_decay': 1e-5,
        'dropout_rate': 0.5,
        'gradient_clip': 1.0,
        'num_epochs': int(epochs),
        'patience': 3,
        'seed': 42,
        'remove_stopwords': False,
        'model_type': model_type,
        'use_pretrained': use_pretrained,
        'use_attention': use_attention,
        'notes': f"Model: {model_type.upper()}, " +
                 f"Embeddings: {'Pretrained' if use_pretrained else 'Scratch'}, " +
                 f"Attention: {'Yes' if use_attention else 'No'}"
    }

    return config


def display_experiment_summary(config):
    """Display summary of experiment configuration."""
    console.print(Panel(
        f"[bold cyan]Experiment Configuration:[/]\n\n"
        f"[yellow]Model:[/] {config['model_type'].upper()}\n"
        f"[yellow]Word Embeddings:[/] {'Pretrained (GloVe)' if config['use_pretrained'] else 'Scratch (Random)'}\n"
        f"[yellow]Attention Mechanism:[/] {'Yes' if config['use_attention'] else 'No'}\n"
        f"[yellow]Batch Size:[/] {config['batch_size']}\n"
        f"[yellow]Learning Rate:[/] {config['learning_rate']}\n"
        f"[yellow]Max Epochs:[/] {config['num_epochs']}\n"
        f"[yellow]Early Stopping Patience:[/] {config['patience']} epochs\n",
        title="Ready to Run",
        border_style="green",
        box=ROUNDED
    ))

    proceed = questionary.confirm("Proceed with this configuration?", default=True).ask()
    return proceed


def run_all_experiments():
    """Run all 8 experiment combinations and display results."""
    console.print(Panel(
        "[bold]Running all 8 combinations of models:[/]\n"
        "- LSTM vs CNN-LSTM\n"
        "- Pretrained vs Scratch Embeddings\n"
        "- With vs Without Attention",
        title="Full Experiment Suite",
        border_style="yellow"
    ))

    proceed = questionary.confirm("This will take some time. Proceed?", default=True).ask()
    if not proceed:
        return

    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)

    # Base configuration
    base_config = {
        'data_path': 'sentiment_data.csv',
        'embedding_path': 'glove.6B.100d.txt',
        'vocab_size': 5000,
        'embedding_dim': 100,
        'hidden_dim': 128,
        'output_dim': 3,  # Positive, Negative, Neutral
        'filter_sizes': [2, 3, 4],
        'num_filters': 64,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'dropout_rate': 0.5,
        'gradient_clip': 1.0,
        'num_epochs': 10,
        'patience': 3,
        'seed': 42,
        'remove_stopwords': False
    }

    results = []

    # Check if data exists, generate if not
    if not os.path.exists(base_config['data_path']):
        with console.status("[bold green]Generating sample data...", spinner="dots"):
            generate_sample_data(base_config['data_path'])

    with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
    ) as progress:
        task = progress.add_task("[cyan]Running experiments...", total=8)

        # Run all eight experiments
        for model_type in ['lstm', 'cnn_lstm']:
            for use_pretrained in [False, True]:
                for use_attention in [False, True]:
                    config = base_config.copy()
                    config['model_type'] = model_type
                    config['use_pretrained'] = use_pretrained
                    config['use_attention'] = use_attention
                    config[
                        'name'] = f"{model_type.upper()}_{'Pretrained' if use_pretrained else 'Scratch'}_{'Attn' if use_attention else 'NoAttn'}"
                    config['save_path'] = f"models/{config['name']}.pt"
                    config['notes'] = f"Model: {model_type.upper()}, " + \
                                      f"Embeddings: {'Pretrained' if use_pretrained else 'Scratch'}, " + \
                                      f"Attention: {'Yes' if use_attention else 'No'}"

                    progress.update(task, description=f"Running {config['name']}...")
                    result = run_experiment(config)
                    results.append(result)
                    progress.advance(task)

    # Display results table
    display_results_table(results)

    # Save results to CSV
    results_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    console.print(f"\nResults saved to [bold green]{results_path}[/]")


def display_results_table(results):
    """Display experiment results in a formatted table."""
    table = Table(title="Experiment Results", box=ROUNDED)

    table.add_column("Model", style="cyan")
    table.add_column("Embeddings", style="green")
    table.add_column("Attention", style="yellow")
    table.add_column("Accuracy", style="magenta")
    table.add_column("F1 Score", style="blue")
    table.add_column("Best Epoch", style="red")

    for result in results:
        experiment = result['experiment']
        parts = experiment.split('_')

        model = parts[0]
        embeddings = "Pretrained" if "Pretrained" in experiment else "Scratch"
        attention = "Yes" if "Attn" in experiment and not "NoAttn" in experiment else "No"

        acc = f"{result['accuracy'] * 100:.2f}%"
        f1 = f"{result['f1_score']:.3f}"
        best_epoch = str(result['best_epoch']) if 'best_epoch' in result else "N/A"

        table.add_row(model, embeddings, attention, acc, f1, best_epoch)

    console.print(table)


def predict_sentiment():
    """Interactive sentiment prediction for user input text."""
    console.print(Panel(
        "[bold]Predict sentiment for your own text and context[/]",
        title="Sentiment Predictor",
        border_style="magenta"
    ))

    # Ask which model to use
    model_files = [f for f in os.listdir('models') if f.endswith('.pt')]
    if not model_files:
        console.print("[bold red]No trained models found. Please train a model first.[/]")
        return

    model_path = questionary.select(
        "Select a trained model to use:",
        choices=model_files
    ).ask()

    # Get text and context
    text = questionary.text("Enter text (max 50 words):").ask()
    context = questionary.text("Enter context (max 20 words):").ask()

    # For now, just display a placeholder result
    # In a real implementation, you would load the model and do inference
    sentiment_map = {0: "Positive", 1: "Negative", 2: "Neutral"}
    prediction = sentiment_map[0]  # Placeholder
    confidence = 0.95  # Placeholder

    console.print(Panel(
        f"[bold]Text:[/] {text}\n"
        f"[bold]Context:[/] {context}\n\n"
        f"[bold]Predicted Sentiment:[/] [bold {get_sentiment_color(prediction)}]{prediction}[/]\n"
        f"[bold]Confidence:[/] {confidence:.2%}",
        title="Prediction Result",
        border_style="green"
    ))


def get_sentiment_color(sentiment):
    """Return color based on sentiment."""
    if sentiment == "Positive":
        return "green"
    elif sentiment == "Negative":
        return "red"
    else:  # Neutral
        return "yellow"


def main_menu():
    """Display the main menu and handle user choices."""
    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                {"name": "Train a single model", "value": "single"},
                {"name": "Run all experiment combinations", "value": "all"},
                {"name": "Predict sentiment for your own text", "value": "predict"},
                {"name": "Exit", "value": "exit"}
            ]
        ).ask()

        if choice == "single":
            config = get_experiment_config()
            if display_experiment_summary(config):
                result = run_experiment(config)

                # Display result
                console.print(Panel(
                    f"[bold]Experiment Results[/]\n\n"
                    f"[yellow]Accuracy:[/] {result['accuracy'] * 100:.2f}%\n"
                    f"[yellow]F1 Score:[/] {result['f1_score']:.3f}\n"
                    f"[yellow]Best Model Saved At:[/] {result['model_path']}\n"
                    f"[yellow]Best Epoch:[/] {result['best_epoch']}",
                    title="Experiment Complete",
                    border_style="green",
                    box=ROUNDED
                ))

        elif choice == "all":
            run_all_experiments()

        elif choice == "predict":
            predict_sentiment()

        elif choice == "exit":
            console.print("[bold cyan]Thank you for using the Sentiment Analyzer![/]")
            break


def main():
    """Main function to run the UI."""
    try:
        # Check for required packages
        try:
            import questionary
            import rich
        except ImportError:
            print("Missing required packages. Installing...")
            os.system("pip install questionary rich")
            print("Please restart the application.")
            sys.exit(1)

        # Print header
        print_header()

        # Show main menu
        main_menu()

    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()