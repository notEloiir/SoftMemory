# Experimental transformer architecture for handling long context
Very experimental  
On LAMBADA performs slightly better compared to normal GPT2 (61.1% to 60.6% accuracy) 
at the cost of way longer computation time (6x time). (On 1 epoch it's same accuracy, 3x time.)   
Granted it's not optimized, but still - not good for general use case model.  
However - LAMBADA test dataset has an average token count of 340 (using GPT2 tokenizer), 
and this "architecture" was thought of for texts longer than standard context windows.

## Prerequisites
- python  
- poetry `pip install poetry`  
- CUDA 12.4 (if using CUDA)  
- \>9 GiB of RAM
- (recommended) CUDA capable GPU with \>4 GiB of VRAM

## Installation
```commandline
git clone https://github.com/notEloiir/SoftMemory.git
cd SoftMemory
poetry install
```

## Run tests
```commandline
poetry run python tests.py
```
