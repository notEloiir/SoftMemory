import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

from gpt2_lightning import GPT2Lightning
from model import Experimental


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, device, max_length=512):
        self.tokenizer = tokenizer
        self.datapoints = list()
        self.max_length = max_length
        self.device = device

        for text in texts:
            i = 0
            while i + (max_length // 2) < len(text) or i == 0:
                self.make_datapoint(text[i * (max_length // 2):(i + 2) * (max_length // 2)])
                i += (max_length // 2)

    def make_datapoint(self, text_part):
        encoded = self.tokenizer(text_part, return_tensors="pt", max_length=self.max_length, truncation=True)
        input_ids = encoded["input_ids"].squeeze(0).to(self.device)  # [seq_len]
        attention_mask = encoded["attention_mask"].squeeze(0).to(self.device)  # [seq_len]
        labels = input_ids.clone()
        self.datapoints.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]


def test_model(model, prompt, extended_context, max_tokens, temp, top_k, top_p):
    generated_text = model.generate_text(prompt, max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
    print(f"\nUntrained {type(model).__name__} generated text:\n{generated_text}")

    if extended_context is None:
        return

    dataset = TextDataset(extended_context, model.tokenizer, model.device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=15, persistent_workers=True)
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)
    trainer.fit(model, dataloader)

    generated_text = model.generate_text(prompt, max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
    print(f"\nTrained {type(model).__name__} generated text:\n{generated_text}")


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    extended_context = [
'''In the dark forest, lived a great vampire. They loved garlic, the smell of it, the taste of it.
"Oh no, what shall I do if everyone dresses up in garlic!" the bloodsucker said.''',
    ]

    # Set input
    prompt = \
'''Question: Do vampires like garlic?
Answer:'''

    # Set params
    max_tokens = 50  # Number of tokens to generate
    temperature = 1.0
    top_k = 50
    top_p = 0.95
    model_name = "gpt2"  # "gpt2", "gpt-medium", "gpt2-large" or "gpt2-xl"

    # Show examples
    mp.set_start_method("spawn")
    test_model(GPT2Lightning(pretrained_model_name=model_name, device=device),
               prompt, extended_context, max_tokens, temperature, top_k, top_p)
    test_model(Experimental(pretrained_model_name=model_name, weighted_mean_init=0.0, device=device),
               prompt, extended_context, max_tokens, temperature, top_k, top_p)
