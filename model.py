import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMSentimentModel(nn.Module):
    """
    LSTM-only model for sentiment analysis.
    Option 1: Simpler architecture.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 pad_idx=0, dropout_rate=0.5, pretrained_embeddings=None, use_attention=False):
        super().__init__()

        # Embeddings
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                padding_idx=pad_idx,
                freeze=False  # Allow fine-tuning
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layers for text and context
        self.text_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.context_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Attention mechanism (optional)
        self.use_attention = use_attention
        if use_attention:
            self.text_attention = Attention(hidden_dim)
            self.context_attention = Attention(hidden_dim)

        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for concatenated text and context
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, context):
        # Embed text and context
        text_embedded = self.embedding(text)  # [batch_size, text_length, embedding_dim]
        context_embedded = self.embedding(context)  # [batch_size, context_length, embedding_dim]

        # Process through LSTM
        text_output, (text_hidden, _) = self.text_lstm(text_embedded)
        context_output, (context_hidden, _) = self.context_lstm(context_embedded)

        # Apply attention if enabled
        if self.use_attention:
            text_vector = self.text_attention(text_output)
            context_vector = self.context_attention(context_output)
        else:
            # Use last hidden state
            text_vector = text_hidden[-1, :, :]  # [batch_size, hidden_dim]
            context_vector = context_hidden[-1, :, :]  # [batch_size, hidden_dim]

        # Combine text and context representations
        combined = torch.cat((text_vector, context_vector), dim=1)
        combined = self.dropout(combined)

        # Final classification
        return self.fc(combined)


class CNNLSTMSentimentModel(nn.Module):
    """
    CNN+LSTM model for sentiment analysis.
    Option 2: Full requirement.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 filter_sizes=[2, 3, 4], num_filters=64, pad_idx=0,
                 dropout_rate=0.5, pretrained_embeddings=None, use_attention=False):
        super().__init__()

        # Embeddings
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                padding_idx=pad_idx,
                freeze=False  # Allow fine-tuning
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # CNN layers for feature extraction
        self.text_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])

        self.context_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])

        # LSTM layers for sequential processing
        self.text_lstm = nn.LSTM(num_filters * len(filter_sizes), hidden_dim, batch_first=True)
        self.context_lstm = nn.LSTM(num_filters * len(filter_sizes), hidden_dim, batch_first=True)

        # Attention mechanism (optional)
        self.use_attention = use_attention
        if use_attention:
            self.text_attention = Attention(hidden_dim)
            self.context_attention = Attention(hidden_dim)

        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, context):
        # Embed text and context
        text_embedded = self.embedding(text)  # [batch_size, text_length, embedding_dim]
        context_embedded = self.embedding(context)  # [batch_size, context_length, embedding_dim]

        # CNN feature extraction (requires permutation for conv1d)
        text_embedded = text_embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, text_length]
        context_embedded = context_embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, context_length]

        # Apply convolutions and max-pooling
        text_conv_outputs = []
        for conv in self.text_convs:
            text_conv_out = F.relu(conv(text_embedded))
            text_conv_out = F.max_pool1d(text_conv_out, text_conv_out.shape[2])
            text_conv_outputs.append(text_conv_out.squeeze(2))

        context_conv_outputs = []
        for conv in self.context_convs:
            context_conv_out = F.relu(conv(context_embedded))
            context_conv_out = F.max_pool1d(context_conv_out, context_conv_out.shape[2])
            context_conv_outputs.append(context_conv_out.squeeze(2))

        # Combine CNN outputs
        text_cnn_features = torch.cat(text_conv_outputs, dim=1).unsqueeze(1)  # Add sequence dimension
        context_cnn_features = torch.cat(context_conv_outputs, dim=1).unsqueeze(1)  # Add sequence dimension

        # Process through LSTM
        text_output, (text_hidden, _) = self.text_lstm(text_cnn_features)
        context_output, (context_hidden, _) = self.context_lstm(context_cnn_features)

        # Apply attention if enabled
        if self.use_attention:
            text_vector = self.text_attention(text_output)
            context_vector = self.context_attention(context_output)
        else:
            # Use last hidden state
            text_vector = text_hidden[-1, :, :]  # [batch_size, hidden_dim]
            context_vector = context_hidden[-1, :, :]  # [batch_size, hidden_dim]

        # Combine text and context representations
        combined = torch.cat((text_vector, context_vector), dim=1)
        combined = self.dropout(combined)

        # Final classification
        return self.fc(combined)


class Attention(nn.Module):
    """
    Attention mechanism for weighted combination of LSTM outputs.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        attn_weights = F.softmax(self.attention(lstm_output).squeeze(2), dim=1)
        # Apply attention weights
        weighted = torch.bmm(attn_weights.unsqueeze(1), lstm_output)
        return weighted.squeeze(1)  # [batch_size, hidden_dim]


def load_pretrained_embeddings(word_to_idx, embedding_path, embedding_dim=100):
    """
    Load GloVe embeddings for the words in our vocabulary.
    """
    # Initialize embeddings with random values
    embeddings = np.random.randn(len(word_to_idx), embedding_dim)

    # Set padding token to all zeros
    embeddings[0] = np.zeros(embedding_dim)

    # Load GloVe
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word in word_to_idx:
                vector = np.array(values[1:], dtype=np.float32)
                idx = word_to_idx[word]
                embeddings[idx] = vector

    return torch.FloatTensor(embeddings)