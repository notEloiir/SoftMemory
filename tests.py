import torch
from torch.utils.data import DataLoader, Dataset

from gpt2_lightning import GPT2Lightning
from model import Experimental


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, device, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], return_tensors="pt", max_length=self.max_length, truncation=True)
        input_ids = encoded["input_ids"].squeeze(0).to(self.device)  # [seq_len]
        attention_mask = encoded["attention_mask"].squeeze(0).to(self.device)  # [seq_len]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_model(model: GPT2Lightning, prompt, extended_context, max_tokens, temp, top_k, top_p):
    generated_text = model.generate_text(prompt, max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
    print(f"\nUntrained {type(model).__name__} generated text:\n{generated_text}")

    if extended_context is None:
        return

    dataset = TextDataset(extended_context, model.tokenizer, model.device)
    dataloader = DataLoader(dataset, batch_size=1)
    model.train_soft(dataloader)
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

    # Show examples
    test_model(GPT2Lightning(device=device), prompt, None, max_tokens, temperature, top_k, top_p)
    test_model(Experimental(weighted_mean_init=0.0, device=device), prompt, extended_context, max_tokens, temperature, top_k, top_p)
