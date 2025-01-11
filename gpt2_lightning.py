import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import pytorch_lightning as pl
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


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

    def training_step(self, batch, batch_idx):
        chunks = batch["chunks"]
        loss = 0

        for chunk in chunks:
            input_ids, attention_mask, labels = chunk["input_ids"], chunk["attention_mask"], chunk["labels"]

            logits = self(input_ids, attention_mask=attention_mask)
            loss += F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        loss /= len(chunks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        chunk = batch["chunks"][-1]
        input_ids, attention_mask = chunk["input_ids"], chunk["attention_mask"]
        label = input_ids[-1]

        logits = self(input_ids, attention_mask=attention_mask)
        predicted_token = torch.argmax(logits[:, -2, :], dim=-1)  # -2 because -1 is the label

        correct = (predicted_token == label).float()
        accuracy = correct.mean()

        self.log("val_accuracy", accuracy, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9, pad_token_id=None, eos_token_id=None):
        generated = input_ids
        for _ in range(max_length):
            # Forward pass through the model to get logits
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # Sample the next token
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)

            # Append the generated token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Stop if eos_token_id is generated
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def generate_text(self, input_text, max_length=50, temperature=1.0, top_k=50, top_p = 0.9):
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(self.device)
        generated_ids = self.generate(input_ids, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
