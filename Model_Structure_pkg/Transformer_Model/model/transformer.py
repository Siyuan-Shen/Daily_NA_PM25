import torch 
from torch import nn

from Model_Structure_pkg.Transformer_Model.model.encoder import Encoder
from Model_Structure_pkg.Transformer_Model.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, input_dim, trg_dim,d_model, n_head, ffn_hidden, num_layers, max_len=1000, drop_prob=0.1, device=None):
        ## input_dim: Dimension of the input features; number of channels
        ## trg_dim: Dimension of the target features; default is 1 for PM2.5
        ## d_model: Dimension of the model (hidden size); default is 64
        ## n_head: Number of attention heads; default is 8
        ## ffn_hidden: Dimension of the feed-forward network hidden layer; default is 256
        ## num_layers: Number of encoder/decoder layers; default is 6
        ## max_len: Maximum length of the input sequence; default is 1000
        ## drop_prob: Dropout probability; default is 0.1
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.trg_dim = trg_dim
        self.d_model = d_model
        self.n_head = n_head
        self.ffn_hidden = ffn_hidden
        self.num_layers = num_layers
        self.max_len = max_len
        self.drop_prob = drop_prob
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(input_dim, d_model, n_head, ffn_hidden, num_layers, max_len=max_len, drop_prob=drop_prob, device=device)
        self.decoder = Decoder(trg_dim, d_model, n_head, ffn_hidden, num_layers, max_len=max_len, drop_prob=drop_prob, device=device)
        self.device = device
        print("Transformer initialized with parameters:")
        print(f"Input Dimension: {input_dim}, Target Dimension: {trg_dim}, d_model: {d_model}, n_head: {n_head}, ffn_hidden: {ffn_hidden}, num_layers: {num_layers}, max_len: {max_len}, drop_prob: {drop_prob}")
        
    
    def forward(self, src_input, target_input=None):

        filled_target_input = torch.nan_to_num(target_input, nan=0.0) if target_input is not None else None
        src_mask = self.make_src_mask(src_input) if src_input is not None else None
        enc_output = self.encoder(src_input, src_mask)
        if target_input is not None:
            # --- SHIFT RIGHT ---
            # dec_in: [B, T, D], first step is BOS (zeros), rest are previous ground truth
            dec_in = torch.zeros_like(filled_target_input)
            dec_in[:, 1:, :] = filled_target_input[:, :-1, :]

            target_mask     = self.make_trg_mask(dec_in)                          # (B,1,T,T)
            enc_dec_mask = self.make_cross_mask(src_input, dec_in.size(1))     # (B,1,T,S)
            # Inside inference loop
            dec_output = self.decoder(dec_in, enc_output, target_mask, enc_dec_mask)
        else:
            # Inference mode: autoregressive decoding
            batch_size = src_input.size(0)
            trg_dim = self.trg_dim
            max_len = src_input.size(1) 
            dec_output = []
            decoder_input = torch.zeros(batch_size, 1, trg_dim, device=self.device)

            for t in range(max_len):
                #print('t:', t)
                #print('size of decoder_input:', decoder_input.size())
                target_mask = self.make_trg_mask(decoder_input)
                enc_dec_mask = self.make_cross_mask(src_input, decoder_input.size(1))
                #enc_dec_mask = src_mask[:, :, 0:decoder_input.size(1), :]
                out_step = self.decoder(decoder_input, enc_output, target_mask, enc_dec_mask)
                next_token = out_step[:, -1:, :]  # last predicted step
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                dec_output.append(next_token)

            dec_output = torch.cat(dec_output, dim=1)
        return dec_output

    def make_cross_mask(self, src_input, trg_len):
        # valid over encoder time: (B,S)
        src_valid = ~torch.isnan(src_input).any(dim=-1)
        # broadcast to (B,1,T,S)
        cross = src_valid[:, None, None, :].expand(-1, 1, trg_len, -1)
        return cross.to(self.device)

    def make_src_mask(self, src_input):
        """
        src_input: (batch_size, src_len, input_dim)
        Returns mask: (batch_size, 1, src_len, src_len)
        scores:  (batch_size, num_heads, seq_len, seq_len)
        mask:    (batch_size, 1, seq_len, seq_len)   ← this broadcasts correctly
        """
        valid_mask = ~torch.isnan(src_input).any(dim=-1)  # (batch_size, src_len)
        src_mask = valid_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
        src_mask = src_mask & src_mask.transpose(-1, -2)  # (batch_size, 1, src_len, src_len)

        return src_mask.to(self.device)
    
    def make_trg_mask(self, target_input):
        """
        target_input: (batch_size, trg_len, output_dim)
        Returns mask: (batch_size, 1, trg_len, trg_len)
        """
        batch_size, trg_len, _ = target_input.size()
        valid_mask = ~torch.isnan(target_input).any(dim=-1)  # (batch_size, trg_len)
        valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)    # (batch_size, 1, 1, trg_len)

        trg_len = target_input.size(1)
        causal_mask = torch.triu(torch.ones(trg_len, trg_len, device=self.device,dtype=torch.bool), diagonal=1)
        causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, trg_len, trg_len)

        trg_mask = valid_mask & valid_mask.transpose(-1, -2) & causal_mask  # (batch_size, 1, trg_len, trg_len)
        return trg_mask.to(self.device)