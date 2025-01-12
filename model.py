import copy

import torch
import torch.nn as nn

from gpt2_lightning import GPT2Lightning


class Experimental(GPT2Lightning):
    def __init__(self, pretrained_model_name="gpt2", weighted_mean_init=0.01):
        super().__init__(pretrained_model_name)

        # Left side (soft)
        self.left = copy.deepcopy(self.h)
        # Right side (hard)
        self.right = self.h  # no deepcopy

        # Freeze parameters in self.right
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze parameters in self.left
        for param in self.left.parameters():
            param.requires_grad = True

        # Weighted mean parameters for each layer
        self.weighted_mean = nn.ParameterList([
            nn.Parameter(torch.rand(self.config.n_embd) * weighted_mean_init, requires_grad=True)
            for _ in range(self.config.n_layer)
        ])

    def configure_optimizers(self):
        return torch.optim.AdamW([
            {'params': self.left.parameters(), 'lr': 1e-3},
            {'params': self.weighted_mean.parameters(), 'lr': 1e-3}
        ])

    def forward(self, input_ids, attention_mask=None):
        # Embedding lookup
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        # Generate position ids if not provided
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Input embeddings
        inputs_embeds = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds)

        # Left (soft) blocks
        left_hidden_states = hidden_states
        for block in self.left:
            left_hidden_states = block(left_hidden_states, attention_mask=attention_mask)[0]

        for block_idx, block in enumerate(self.right):
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

            # Weighted Mean Layer
            residual = hidden_states
            weighted_left_states = left_hidden_states * self.weighted_mean[block_idx].unsqueeze(0).unsqueeze(1)
            hidden_states = residual + weighted_left_states

        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        return logits

    def reset_soft(self):
        self.left = copy.deepcopy(self.h)
