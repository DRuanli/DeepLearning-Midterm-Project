import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class SentimentDataset(Dataset):
    def __init__(self, texts, contexts, labels, word_to_idx, max_len_text=50, max_len_context=20):
        self.texts = texts
        self.contexts = contexts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len_text = max_len_text
        self.max_len_context = max_len_context

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        context = self.contexts[idx]
        label = self.labels[idx]

        # Convert text and context to indices
        text_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in text.split()]
        context_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in context.split()]

        # Padding or truncating
        if len(text_indices) < self.max_len_text:
            text_indices += [self.word_to_idx['<PAD>']] * (self.max_len_text - len(text_indices))
        else:
            text_indices = text_indices[:self.max_len_text]

        if len(context_indices) < self.max_len_context:
            context_indices += [self.word_to_idx['<PAD>']] * (self.max_len_context - len(context_indices))
        else:
            context_indices = context_indices[:self.max_len_context]

        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'context': torch.tensor(context_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def build_vocabulary(texts, contexts, max_vocab_size=5000):
    """Build a vocabulary from all texts and contexts."""
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    for context in contexts:
        all_words.extend(context.split())

    # Count word frequencies
    word_counts = Counter(all_words)

    # Sort by frequency and limit to max_vocab_size
    most_common = word_counts.most_common(max_vocab_size - 3)  # Reserve space for <PAD>, <UNK>, etc.

    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def load_data(csv_path, max_vocab_size=5000, batch_size=32, test_size=0.2, random_state=42):
    """Load data from CSV, preprocess, and create DataLoaders."""
    df = pd.read_csv(csv_path)

    # Check required columns are present
    if not all(col in df.columns for col in ['text', 'context', 'label']):
        raise ValueError("CSV file must contain 'text', 'context', and 'label' columns")

    # Map text labels to integers if needed
    if not pd.api.types.is_numeric_dtype(df['label']):
        label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        df['label'] = df['label'].map(label_map)

    # Preprocess texts and contexts
    df['text'] = df['text'].apply(preprocess_text)
    df['context'] = df['context'].apply(preprocess_text)

    # Handle missing values
    df = df.dropna()

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])

    # Build vocabulary from training data only
    word_to_idx = build_vocabulary(train_df['text'], train_df['context'], max_vocab_size)

    # Create datasets
    train_dataset = SentimentDataset(
        train_df['text'].values,
        train_df['context'].values,
        train_df['label'].values,
        word_to_idx
    )

    test_dataset = SentimentDataset(
        test_df['text'].values,
        test_df['context'].values,
        test_df['label'].values,
        word_to_idx
    )

    # Calculate class weights for handling imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, word_to_idx, class_weights


def generate_sample_data(output_path='sentiment_data.csv', num_samples=500):
    """Generate a sample dataset for testing."""
    # Sample texts, contexts, and corresponding labels
    sample_data = []

    # Positive samples
    positive_texts = [
        "I love this new feature!", "The product exceeded my expectations.",
        "Great improvement to the service.", "This is the best solution available.",
        "I'm extremely happy with the results.", "Excellent customer support experience.",
        "The team delivered outstanding work.", "This makes my job so much easier.",
        "I'm impressed by the quality.", "The update fixed all my issues."
    ]

    positive_contexts = [
        "After the upgrade.", "Compared to competitors.",
        "Used it for a month.", "Following the latest release.",
        "After speaking with support.", "During the trial period.",
        "When working on a major project.", "After the training session.",
        "When reviewing the specs.", "After the bug fix."
    ]

    # Negative samples
    negative_texts = [
        "This doesn't work at all.", "I'm disappointed with the service.",
        "Too many bugs in this version.", "The interface is confusing.",
        "Customer support was unhelpful.", "This is worse than before.",
        "I regret purchasing this product.", "Many features are broken.",
        "The performance is terrible.", "This caused more problems."
    ]

    negative_contexts = [
        "After the recent update.", "During peak usage hours.",
        "When trying to complete my work.", "After paying for premium.",
        "When dealing with support.", "During the implementation phase.",
        "When showing to clients.", "After multiple attempts.",
        "During the deadline.", "Under normal conditions."
    ]

    # Neutral samples
    neutral_texts = [
        "It works as expected.", "The features are standard.",
        "It's similar to the previous version.", "Does what it says.",
        "Average performance overall.", "Some things work, some don't.",
        "It's okay for the price.", "Neither impressive nor disappointing.",
        "It gets the job done.", "Basic functionality is there."
    ]

    neutral_contexts = [
        "For everyday use.", "According to the manual.",
        "As described in docs.", "For routine tasks.",
        "During normal operation.", "For most users.",
        "In standard environments.", "For typical workloads.",
        "In most situations.", "For regular activities."
    ]

    # Generate data by mixing and matching
    for _ in range(num_samples // 3):
        # Positive samples
        sample_data.append({
            'text': np.random.choice(positive_texts),
            'context': np.random.choice(positive_contexts),
            'label': 'Positive'
        })

        # Negative samples
        sample_data.append({
            'text': np.random.choice(negative_texts),
            'context': np.random.choice(negative_contexts),
            'label': 'Negative'
        })

        # Neutral samples
        sample_data.append({
            'text': np.random.choice(neutral_texts),
            'context': np.random.choice(neutral_contexts),
            'label': 'Neutral'
        })

    # Create and save dataframe
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset with {len(df)} examples saved to {output_path}")

    return output_path


# If run directly, generate sample data
if __name__ == "__main__":
    generate_sample_data(num_samples=510)  # Generate slightly more than 500 samples