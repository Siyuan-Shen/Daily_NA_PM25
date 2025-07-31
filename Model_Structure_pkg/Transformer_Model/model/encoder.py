from torch import nn

from Model_Structure_pkg.Transformer_Model.embeddings.Sequence_Embedding import Sequence_Embedding
from Model_Structure_pkg.Transformer_Model.blocks.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_head, ffn_hidden, num_layers, max_len=1000, drop_prob=0.1, device=None):
        super(Encoder, self).__init__()
        self.embedding = Sequence_Embedding(input_dim, d_model, max_len=max_len, drop_prob=drop_prob, device=device)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_head, ffn_hidden, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x, src_mask=None):
        x = self.embedding(x)  # Apply embedding layer
        for layer in self.layers:

            x = layer(x, src_mask)
        return x