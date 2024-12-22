import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
import pytorch_lightning as pl
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


# Test model that mirrors GPT2LMHeadModel.from_pretrained(pretrained_model_name)
# Used to compare sanity of experimental model
class GPT2Lightning(pl.LightningModule):
    def __init__(self, pretrained_model_name="gpt2"):
        super().__init__()
        # Load the GPT2 configuration and pretrained weights
        self.config = GPT2Config.from_pretrained(pretrained_model_name)
        self.config.add_cross_attention = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)

        # Define the GPT2 model components
        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)  # Word embeddings
        self.wpe = nn.Embedding(self.config.n_positions, self.config.n_embd)  # Positional embeddings
        self.drop = nn.Dropout(self.config.embd_pdrop)  # Dropout layer
        self.h = nn.ModuleList([GPT2Block(self.config) for _ in range(self.config.n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(self.config.n_embd, eps=self.config.layer_norm_epsilon)  # Final layer norm
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)  # Language modeling head

        # Load the pretrained weights
        self.load_pretrained_weights(pretrained_model_name)

    def load_pretrained_weights(self, pretrained_model_name):
        pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        state_dict = pretrained_model.state_dict()

        # Adjust the keys to match the custom model's structure
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                new_key = key[len("transformer."):]  # Remove the "transformer." prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load the adjusted state_dict
        self.load_state_dict(new_state_dict)

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

        # Transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        return logits

    '''
    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self.forward(input_ids)
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        self.log("train_loss", loss)
        return loss
    '''

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

    def generate_text(self, input_text, max_tokens, device):
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)

        generated_ids = input_ids.clone()
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                output = self(generated_ids)
                # Handle model-specific output format
                logits = output[:, -1, :]
                # Predict the next token
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        # Decode the sequence
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
