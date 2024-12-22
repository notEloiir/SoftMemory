import torch
import torch.nn as nn

from gpt2_lightning import GPT2Lightning


class Experimental(GPT2Lightning):
    def __init__(self, pretrained_model_name="gpt2", weighted_mean_init=0.01):
        super().__init__(pretrained_model_name)

        # Left side (soft)
        self.left = self.h  # TODO: this should be a deepcopy
        # Right side (hard)
        self.right = self.h  # no deepcopy

        # Weighted mean parameters for each layer
        self.weighted_mean = nn.ParameterList([nn.Parameter(weighted_mean_init * torch.ones(self.config.n_embd))
                                               for _ in range(self.config.n_layer)])

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

        for layer_idx in range(self.config.n_layer):
            layer = self.right[layer_idx]

            # modified GPT2Block.forward

            # GPT-2 Layer: Self-Attention + Add & Norm
            residual = hidden_states
            hidden_states = layer.ln_1(hidden_states)
            attn_outputs = layer.attn(hidden_states, attention_mask=attention_mask, layer_past=None, use_cache=False, output_attentions=False)
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            # residual connection
            hidden_states = attn_output + residual

            # Weighted Mean Layer
            residual = hidden_states
            weighted_left_states = left_hidden_states * self.weighted_mean[layer_idx].unsqueeze(0).unsqueeze(1)
            hidden_states = residual + weighted_left_states

            residual = hidden_states
            hidden_states = layer.ln_2(hidden_states)
            feed_forward_hidden_states = layer.mlp(hidden_states)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states

            outputs = (hidden_states,) + outputs[1:]

            hidden_states = outputs[0]

        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        return logits

    def configure_optimizers(self):
        # TODO: different learning rates for left, weighted mean, right
        return torch.optim.AdamW(self.parameters(), lr=5e-5)
