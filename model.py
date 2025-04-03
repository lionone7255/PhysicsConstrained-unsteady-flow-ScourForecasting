import torch.nn as nn


class PhysicsInformedTransformer(nn.Module):
    """Transformer architecture with physical constraints integration"""

    def __init__(self, model_config):
        super().__init__()
        self.embedding = nn.Linear(model_config['input_dim'],
                                   model_config['hidden_dim'])
        self.positional_encoding = self._init_positional_encoding(model_config)

    def _init_positional_encoding(self, config):
        """Initialize temporal encoding scheme"""
        return nn.Embedding(config['max_seq_len'], config['hidden_dim'])

    def forward(self, x):
        """Main forward pass with physics constraints"""
        x = self._apply_embeddings(x)
        return self._temporal_processing(x)