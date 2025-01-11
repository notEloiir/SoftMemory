import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from datasets import load_dataset

from gpt2_lightning import GPT2Lightning
from model import Experimental


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, device=torch.device("cpu")):
        # if using lightning (as it should be), device parameter should be left at default
        self.tokenizer = tokenizer
        self.datapoints = list()
        self.max_length = max_length

        for dp in data:
            text = dp["text"]
            tokens = self.tokenizer(text, return_tensors="pt")
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            datapoint = {"chunks": list()}
            start, step = 0, (max_length // 2)
            while start + step < input_ids.size(0) or start == 0:
                end = min(start + max_length, input_ids.size(0))
                datapoint["chunks"].append({
                    "input_ids": input_ids[start : (end - 1)].clone().to(device),
                    "attention_mask": attention_mask[start : (end - 1)].clone().to(device),
                    "labels": input_ids[(start + 1) : end].clone().to(device),
                })
                start += step
            self.datapoints.append(datapoint)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]


def test_model(model, prompt, extended_context, max_tokens, temp, top_k, top_p):
    generated_text = model.generate_text(prompt, max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
    print(f"\nUntrained {type(model).__name__} generated text:\n{generated_text}")

    if extended_context is None:
        return

    dataset = TextDataset(extended_context, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=15, persistent_workers=True)
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)
    trainer.fit(model, dataloader)

    generated_text = model.generate_text(prompt, max_tokens, temperature=temp, top_k=top_k, top_p=top_p)
    print(f"\nTrained {type(model).__name__} generated text:\n{generated_text}")


def benchmark_model(model, dataset_name="lambada"):
    # this isn't in lightning because of a severe case of skill issue
    # (or that lightning isn't made for custom stuff)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    dataset = TextDataset(load_dataset(dataset_name, split="test"), model.tokenizer, device=device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=15, persistent_workers=True)
    optimizer = model.configure_optimizers()

    print("Benchmark started, please be patient")
    val_accuracy = list()
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(model, Experimental):
            model.reset_soft()

            model.train()
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_accuracy.append(model.validation_step(batch, batch_idx))

        if batch_idx and batch_idx % 500 == 0:
            print(f"Validation Accuracy ({batch_idx} of {len(dataloader)} done): {sum(val_accuracy) / batch_idx:.2%}")
    print(f"Validation Accuracy (all done): {sum(val_accuracy) / len(dataloader):.2%}")


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    extended_context = [dict()]
    extended_context[0]["text"] = \
'''In the dark forest, lived a great vampire. They loved garlic, the smell of it, the taste of it.
"Oh no, what shall I do if everyone dresses up in garlic!" the bloodsucker said.'''

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
    weighted_mean_init = 0.01

    # Show examples
    mp.set_start_method("spawn")
    # test_model(GPT2Lightning(pretrained_model_name=model_name),
    #            prompt, extended_context, max_tokens, temperature, top_k, top_p)
    # test_model(Experimental(pretrained_model_name=model_name, weighted_mean_init=weighted_mean_init),
    #            prompt, extended_context, max_tokens, temperature, top_k, top_p)

    # Benchmark
    # normal GPT2-small has 60.53% accuracy
    # benchmark_model(GPT2Lightning(pretrained_model_name=model_name))
    # experimental model based on GPT2-small has 59.23% accuracy so far
    benchmark_model(Experimental(pretrained_model_name=model_name, weighted_mean_init=weighted_mean_init))
