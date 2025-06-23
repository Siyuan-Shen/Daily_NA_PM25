import torch 
from torch import nn

from Model_Structure_pkg.Transformer_Model.model.encoder import Encoder
from Model_Structure_pkg.Transformer_Model.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, input_dim, trg_dim,d_model, n_head, ffn_hidden, num_layers, max_len=1000, drop_prob=0.1, device=None):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_dim, d_model, n_head, ffn_hidden, num_layers, max_len=max_len, drop_prob=drop_prob, device=device)
        self.decoder = Decoder(trg_dim, d_model, n_head, ffn_hidden, num_layers, max_len=max_len, drop_prob=drop_prob, device=device)
        self.device = device
        
    def forward(self, src_input, target_input=None):
        src_mask = self.make_src_mask(target_input)
        target_mask = self.make_trg_mask(target_input)
        enc_output = self.encoder(src_input, src_mask)
        dec_output = self.decoder(target_input, enc_output, target_mask, src_mask)
        return dec_output
    
    def make_src_mask(self, target_input):
        batchsize, trg_len = target_input.size()
        # Create a target mask to prevent attending to future tokens
        valid_mask = ~torch.isnan(target_input)
        # Create a source mask to prevent attending to padding tokens
        src_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2) 
        return src_mask.to(self.device)
    
    def make_trg_mask(self, target_input):
        batchsize, trg_len = target_input.size()
        valid_mask = ~torch.isnan(target_input)  # (batch_size, seq_len), True = valid
        casual_mask = torch.triu(torch.ones((trg_len, trg_len), device=self.device), diagonal=1).bool()  # Upper triangular matrix
        trg_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2) & ~casual_mask  # (batch_size, trg_len, trg_len)
        return trg_mask.to(self.device)