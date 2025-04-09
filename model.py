import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMSentimentModel(nn.Module):
    """
    LSTM-only model for sentiment analysis with bidirectional LSTM.
    Option 1: Simpler architecture with improvements.
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

        # Embedding dropout
        self.emb_dropout = nn.Dropout(0.2)

        # Bidirectional LSTM layers
        self.text_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Half size for bidirectional
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if dropout_rate > 0 else 0
        )

        self.context_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Half size for bidirectional
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if dropout_rate > 0 else 0
        )

        # Attention mechanism (optional)
        self.use_attention = use_attention
        if use_attention:
            self.text_attention = SelfAttention(hidden_dim)
            self.context_attention = SelfAttention(hidden_dim)

        # Classifier with layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, context):
        # Embed text and context with dropout
        text_embedded = self.emb_dropout(self.embedding(text))  # [batch_size, text_length, embedding_dim]
        context_embedded = self.emb_dropout(self.embedding(context))  # [batch_size, context_length, embedding_dim]

        # Process through bidirectional LSTM
        text_output, (text_hidden, _) = self.text_lstm(text_embedded)
        context_output, (context_hidden, _) = self.context_lstm(context_embedded)

        # For bidirectional LSTM, concatenate the last hidden state of both directions
        # text_hidden shape: [2, batch_size, hidden_dim//2]
        text_hidden_forward = text_hidden[0, :, :]
        text_hidden_backward = text_hidden[1, :, :]
        text_final = torch.cat((text_hidden_forward, text_hidden_backward), dim=1)

        context_hidden_forward = context_hidden[0, :, :]
        context_hidden_backward = context_hidden[1, :, :]
        context_final = torch.cat((context_hidden_forward, context_hidden_backward), dim=1)

        # Apply attention if enabled
        if self.use_attention:
            text_vector = self.text_attention(text_output)
            context_vector = self.context_attention(context_output)
        else:
            # Use the concatenated hidden states
            text_vector = text_final
            context_vector = context_final

        # Combine text and context representations with element-wise addition and concatenation
        combined = torch.cat((text_vector, context_vector), dim=1)
        combined = self.layer_norm(combined)
        combined = self.fc_dropout(combined)

        # Two-layer classifier with ReLU activation
        hidden = F.relu(self.fc1(combined))
        hidden = self.fc_dropout(hidden)
        output = self.fc2(hidden)

        return output


class CNNLSTMSentimentModel(nn.Module):
    """
    CNN+LSTM model for sentiment analysis with improvements.
    Option 2: Full requirement with enhanced architecture.
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

        # Embedding dropout
        self.emb_dropout = nn.Dropout(0.2)

        # CNN layers for feature extraction with batch normalization
        self.text_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs, padding=fs // 2)
            for fs in filter_sizes
        ])

        self.text_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in filter_sizes
        ])

        self.context_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs, padding=fs // 2)
            for fs in filter_sizes
        ])

        self.context_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in filter_sizes
        ])

        # Bidirectional LSTM layers
        self.text_lstm = nn.LSTM(
            num_filters * len(filter_sizes),
            hidden_dim // 2,  # Half size for bidirectional
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if dropout_rate > 0 else 0
        )

        self.context_lstm = nn.LSTM(
            num_filters * len(filter_sizes),
            hidden_dim // 2,  # Half size for bidirectional
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if dropout_rate > 0 else 0
        )

        # Attention mechanism (optional)
        self.use_attention = use_attention
        if use_attention:
            self.text_attention = SelfAttention(hidden_dim)
            self.context_attention = SelfAttention(hidden_dim)

        # Improved classifier with layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, context):
        # Embed text and context with dropout
        text_embedded = self.emb_dropout(self.embedding(text))  # [batch_size, text_length, embedding_dim]
        context_embedded = self.emb_dropout(self.embedding(context))  # [batch_size, context_length, embedding_dim]

        # CNN feature extraction (requires permutation for conv1d)
        text_embedded = text_embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, text_length]
        context_embedded = context_embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, context_length]

        # Apply convolutions with batch normalization and ReLU
        text_conv_outputs = []
        for i, (conv, bn) in enumerate(zip(self.text_convs, self.text_batch_norms)):
            text_conv_out = conv(text_embedded)
            text_conv_out = bn(text_conv_out)
            text_conv_out = F.relu(text_conv_out)
            text_conv_out = F.max_pool1d(text_conv_out, text_conv_out.shape[2])
            text_conv_outputs.append(text_conv_out.squeeze(2))

        context_conv_outputs = []
        for i, (conv, bn) in enumerate(zip(self.context_convs, self.context_batch_norms)):
            context_conv_out = conv(context_embedded)
            context_conv_out = bn(context_conv_out)
            context_conv_out = F.relu(context_conv_out)
            context_conv_out = F.max_pool1d(context_conv_out, context_conv_out.shape[2])
            context_conv_outputs.append(context_conv_out.squeeze(2))

        # Combine CNN outputs
        text_cnn_features = torch.cat(text_conv_outputs, dim=1).unsqueeze(1)  # Add sequence dimension
        context_cnn_features = torch.cat(context_conv_outputs, dim=1).unsqueeze(1)  # Add sequence dimension

        # Process through bidirectional LSTM
        text_output, (text_hidden, _) = self.text_lstm(text_cnn_features)
        context_output, (context_hidden, _) = self.context_lstm(context_cnn_features)

        # For bidirectional LSTM, concatenate the final hidden states from both directions
        text_hidden_forward = text_hidden[0, :, :]
        text_hidden_backward = text_hidden[1, :, :]
        text_final = torch.cat((text_hidden_forward, text_hidden_backward), dim=1)

        context_hidden_forward = context_hidden[0, :, :]
        context_hidden_backward = context_hidden[1, :, :]
        context_final = torch.cat((context_hidden_forward, context_hidden_backward), dim=1)

        # Apply attention if enabled
        if self.use_attention:
            text_vector = self.text_attention(text_output)
            context_vector = self.context_attention(context_output)
        else:
            # Use the concatenated hidden states
            text_vector = text_final
            context_vector = context_final

        # Combine text and context representations
        combined = torch.cat((text_vector, context_vector), dim=1)
        combined = self.layer_norm(combined)
        combined = self.fc_dropout(combined)

        # Two-layer classifier with ReLU activation
        hidden = F.relu(self.fc1(combined))
        hidden = self.fc_dropout(hidden)
        output = self.fc2(hidden)

        return output


class SelfAttention(nn.Module):
    """
    Enhanced self-attention mechanism with scaled dot-product attention.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, seq_len, hidden_dim]

        # Project to queries, keys, values
        Q = self.query(hidden_states)  # [batch_size, seq_len, hidden_dim]
        K = self.key(hidden_states)  # [batch_size, seq_len, hidden_dim]
        V = self.value(hidden_states)  # [batch_size, seq_len, hidden_dim]

        # Calculate scaled dot-product attention
        # (batch_size, seq_len, hidden_dim) x (batch_size, hidden_dim, seq_len)
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale  # [batch_size, seq_len, seq_len]

        # Convert to attention weights
        attention = F.softmax(energy, dim=-1)  # [batch_size, seq_len, seq_len]

        # Apply attention to values
        weighted = torch.matmul(attention, V)  # [batch_size, seq_len, hidden_dim]

        # Sum over sequence dimension to get a single vector per batch item
        return weighted.sum(dim=1)  # [batch_size, hidden_dim]


def load_pretrained_embeddings(word_to_idx, embedding_path, embedding_dim=100):
    """
    Load GloVe embeddings for the words in our vocabulary.
    """
    # Initialize embeddings with random values
    embeddings = np.random.normal(scale=0.1, size=(len(word_to_idx), embedding_dim))

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