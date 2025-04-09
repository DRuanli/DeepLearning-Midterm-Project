import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time
import os
import argparse
from tqdm import tqdm

from data import load_data, generate_sample_data
from model import LSTMSentimentModel, CNNLSTMSentimentModel, load_pretrained_embeddings


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        text = batch['text'].to(device)
        context = batch['context'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(text, context)

        # Calculate loss and backward pass
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()

        # Get predictions
        _, predicted = torch.max(predictions, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss / len(train_loader), acc, f1


def evaluate(model, test_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].to(device)
            context = batch['context'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            predictions = model(text, context)

            # Calculate loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(predictions, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss / len(test_loader), acc, f1


def run_experiment(config):
    """Run a single experiment with the given configuration."""
    print(f"\n{'=' * 50}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'=' * 50}")

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    if not os.path.exists(config['data_path']):
        print(f"Data file not found at {config['data_path']}. Generating sample data...")
        generate_sample_data(config['data_path'])

    train_loader, test_loader, word_to_idx, class_weights = load_data(
        config['data_path'],
        max_vocab_size=config['vocab_size'],
        batch_size=config['batch_size'],
        test_size=0.2
    )

    # Load pretrained embeddings if specified
    pretrained_embeddings = None
    if config['use_pretrained']:
        if os.path.exists(config['embedding_path']):
            print(f"Loading pretrained embeddings from {config['embedding_path']}...")
            pretrained_embeddings = load_pretrained_embeddings(
                word_to_idx,
                config['embedding_path'],
                config['embedding_dim']
            )
        else:
            print(f"Warning: Pretrained embeddings path {config['embedding_path']} not found. Using random embeddings.")

    # Initialize model
    if config['model_type'] == 'lstm':
        model = LSTMSentimentModel(
            vocab_size=len(word_to_idx),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            pad_idx=0,
            dropout_rate=config['dropout_rate'],
            pretrained_embeddings=pretrained_embeddings,
            use_attention=config['use_attention']
        )
    else:  # CNN+LSTM
        model = CNNLSTMSentimentModel(
            vocab_size=len(word_to_idx),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            filter_sizes=config['filter_sizes'],
            num_filters=config['num_filters'],
            pad_idx=0,
            dropout_rate=config['dropout_rate'],
            pretrained_embeddings=pretrained_embeddings,
            use_attention=config['use_attention']
        )

    model = model.to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Use class weights for imbalanced data if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    best_valid_f1 = 0
    best_epoch = 0
    train_losses, train_accs, train_f1s = [], [], []
    valid_losses, valid_accs, valid_f1s = [], [], []

    for epoch in range(config['num_epochs']):
        start_time = time.time()

        # Train
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)

        # Evaluate
        valid_loss, valid_acc, valid_f1 = evaluate(model, test_loader, criterion, device)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        # Check for early stopping
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_epoch = epoch

            # Save the model
            torch.save(model.state_dict(), config['save_path'])

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_f1:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}% | Valid F1: {valid_f1:.3f}')

        # Early stopping
        if epoch - best_epoch >= config['patience']:
            print(f"No improvement for {config['patience']} epochs. Early stopping.")
            break

    # Load best model
    model.load_state_dict(torch.load(config['save_path']))

    # Final evaluation
    final_loss, final_acc, final_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Loss: {final_loss:.3f} | Accuracy: {final_acc * 100:.2f}% | F1: {final_f1:.3f}")

    # Return final results
    return {
        'experiment': config['name'],
        'accuracy': final_acc,
        'f1_score': final_f1,
        'notes': config['notes']
    }


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate sentiment analysis models')
    parser.add_argument('--data_path', type=str, default='sentiment_data.csv', help='Path to the CSV data file')
    parser.add_argument('--model_type', type=str, default='cnn_lstm', choices=['lstm', 'cnn_lstm'],
                        help='Model architecture to use')
    parser.add_argument('--run_all', action='store_true', help='Run all four experiment combinations')
    parser.add_argument('--embedding_path', type=str, default='glove.6B.100d.txt',
                        help='Path to pretrained embeddings file')
    parser.add_argument('--use_pretrained', action='store_true', help='Whether to use pretrained embeddings')
    parser.add_argument('--use_attention', action='store_true', help='Whether to use attention mechanism')
    parser.add_argument('--results_path', type=str, default='results.csv', help='Path to save results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Base configuration
    base_config = {
        'data_path': args.data_path,
        'embedding_path': args.embedding_path,
        'save_path': f'models/sentiment_model.pt',
        'vocab_size': 5000,
        'embedding_dim': 100,
        'hidden_dim': 128,
        'output_dim': 3,  # Positive, Negative, Neutral
        'filter_sizes': [2, 3, 4],
        'num_filters': 64,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'num_epochs': 10,
        'patience': 3,
        'seed': 42
    }

    # If run_all, run all four experiment combinations
    if args.run_all:
        experiments = []
        results = []

        # Run all four experiments
        for model_type in ['lstm', 'cnn_lstm']:
            for use_pretrained in [True, False]:
                for use_attention in [True, False]:
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

                    result = run_experiment(config)
                    results.append(result)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.results_path, index=False)
        print(f"\nResults saved to {args.results_path}")

        # Print results table
        print("\nEXPERIMENTS SUMMARY:")
        print(f"{'=' * 80}")
        print(f"| {'Experiment':<20} | {'Accuracy':<10} | {'F1-score':<10} | {'Notes':<30} |")
        print(f"|{'-' * 22}|{'-' * 12}|{'-' * 12}|{'-' * 32}|")

        for result in results:
            exp_name = result['experiment']
            acc = f"{result['accuracy'] * 100:.2f}%"
            f1 = f"{result['f1_score']:.3f}"
            notes = result['notes']
            print(f"| {exp_name:<20} | {acc:<10} | {f1:<10} | {notes:<30} |")

        print(f"{'=' * 80}")

    else:
        # Run single experiment with specified parameters
        config = base_config.copy()
        config['model_type'] = args.model_type
        config['use_pretrained'] = args.use_pretrained
        config['use_attention'] = args.use_attention
        config[
            'name'] = f"{args.model_type.upper()}_{'Pretrained' if args.use_pretrained else 'Scratch'}_{'Attn' if args.use_attention else 'NoAttn'}"
        config['save_path'] = f"models/{config['name']}.pt"
        config['notes'] = f"Model: {args.model_type.upper()}, " + \
                          f"Embeddings: {'Pretrained' if args.use_pretrained else 'Scratch'}, " + \
                          f"Attention: {'Yes' if args.use_attention else 'No'}"

        run_experiment(config)


if __name__ == "__main__":
    main()