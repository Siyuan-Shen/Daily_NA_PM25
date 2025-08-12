from torch import nn
import torch

from Model_Structure_pkg.Transformer_Model.embeddings.Sequence_Embedding import Sequence_Embedding
from Model_Structure_pkg.Transformer_Model.blocks.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, input_dim, d_model, n_head, ffn_hidden, num_layers, max_len=1000, drop_prob=0.1, device=None):
        super(Decoder, self).__init__()
        self.embedding = Sequence_Embedding(input_dim, d_model, max_len=max_len, drop_prob=drop_prob, device=device)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_head, ffn_hidden, drop_prob) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, input_dim)  # Linear layer to project output to input dimension
    
    def forward(self, x, enc_output=None, target_mask=None, src_mask=None):

        #x = x.unsqueeze(-1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_output, target_mask, src_mask)
        x = self.linear(x)
        #x = x.unsqueeze(-1)  # Remove the last dimension
        return x