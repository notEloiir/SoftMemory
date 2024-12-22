import torch

from gpt2_lightning import GPT2Lightning
from model import Experimental


if __name__ == "__main__":
    print("Hello world!\n")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    # Load the models
    gpt2_lightning = GPT2Lightning()
    custom_model = Experimental(weighted_mean_init=0.01)

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_lightning = gpt2_lightning.to(device)
    custom_model = custom_model.to(device)

    # Set input
    input_text = "How are you?"
    max_tokens = 10  # Number of tokens to generate

    # Generate text using both models
    gpt2_lightning_generated_test = gpt2_lightning.generate_text(input_text, max_tokens, device)
    custom_generated_text = custom_model.generate_text(input_text, max_tokens, device)

    # Print the results
    print("\nNormal GPT-2 generated text:")
    print(gpt2_lightning_generated_test)
    print("\nCustom model generated text:")
    print(custom_generated_text)
