import torch

from gpt2_lightning import GPT2Lightning
from model import Experimental


if __name__ == "__main__":
    print("Hello world!\n")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the models
    gpt2_lightning = GPT2Lightning(device=device)
    custom_model = Experimental(weighted_mean_init=0.0, device=device)

    # Set input
    input_text = \
'''In the dark forest, lived a great vampire. They loved garlic, the smell of it, the taste of it.
"Oh no, what shall I do if everyone dresses up in garlic!" the bloodsucker said.
"We should'''
    max_tokens = 50  # Number of tokens to generate

    # Set params
    temperature = 1.0
    top_k = 50
    top_p = 0.95

    # Generate text using both models
    gpt2_lightning_generated_test = gpt2_lightning.generate_text(input_text, max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    custom_generated_text = custom_model.generate_text(input_text, max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)

    # Print the results
    print("\nNormal GPT-2 generated text:")
    print(gpt2_lightning_generated_test)
    print("\nCustom model generated text:")
    print(custom_generated_text)
