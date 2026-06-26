# Nano LLM

A tiny BPE-tokenized language model built with PyTorch. This project trains a small GPT-style transformer on PDF books, saves a checkpoint and tokenizer, then lets you chat with the trained model from the terminal.

The code is intentionally compact and readable, making it useful for learning how tokenization, self-attention, transformer blocks, training loops, checkpoints, and text generation fit together.

## Features

- Extracts text from PDF files in `books/`
- Cleans common PDF encoding artifacts
- Builds a BPE tokenizer from the training corpus
- Trains a small decoder-only transformer inspired by NanoGPT
- Saves the trained model to `checkpoints/nano_llm.pt`
- Saves the tokenizer to `checkpoints/tokenizer.json`
- Provides an interactive chat script for text generation
- Automatically uses CUDA when available

## Project Structure

```text
nano-LLM/
+-- books/                  # PDF training files
+-- checkpoints/            # Saved model and tokenizer
|   +-- nano_llm.pt
|   +-- tokenizer.json
+-- train_nano_llm.py       # Training script and model definition
+-- chat_nano_llm.py        # Interactive inference script
+-- architecture.md         # Short architecture notes
+-- nano.ipynb              # Notebook experiments
+-- README.md
```

## How It Works

The training pipeline follows these steps:

1. Read PDF files from the `books/` directory.
2. Extract and clean text.
3. Train or load a BPE tokenizer.
4. Encode text into token IDs.
5. Create input-target training batches.
6. Train a tiny transformer language model.
7. Save the checkpoint and tokenizer.
8. Generate sample text from the trained model.

The model uses:

- Token embeddings
- Positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward layers
- Layer normalization
- Transformer blocks
- A language modeling head

## Requirements

- Python 3.10 or newer
- PyTorch
- pypdf
- tokenizers

Install dependencies:

```bash
pip install torch pypdf tokenizers
```

If you have a CUDA-enabled GPU, install the PyTorch build that matches your CUDA version from the official PyTorch install page.

## Dataset

Place PDF files inside the `books/` directory:

```text
books/
+-- Alice_in_Wonderland.pdf
+-- Sherlock holmes.pdf
+-- ...
```

The training script reads every `.pdf` file in this folder.

Note: this is a tiny BPE-tokenized model trained on a small local corpus. Its output will imitate patterns from the training text, but it will not behave like a large instruction-tuned chatbot.

## Training

Run the default training command:

```bash
python train_nano_llm.py
```

By default, this trains with:

- Batch size: `32`
- Context length: `128`
- Training steps: `2000`
- Embedding size: `128`
- Attention heads: `4`
- Transformer layers: `4`
- Dropout: `0.2`
- Learning rate: `3e-4`

After training, the script saves:

```text
checkpoints/nano_llm.pt
checkpoints/tokenizer.json
```

### Custom Training Example

```bash
python train_nano_llm.py \
  --books-dir books \
  --out-dir checkpoints \
  --batch-size 32 \
  --block-size 128 \
  --max-iters 3000 \
  --eval-interval 200 \
  --eval-iters 20 \
  --learning-rate 0.0003 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4 \
  --dropout 0.2
```

On Windows PowerShell, use backticks for line continuation:

```powershell
python train_nano_llm.py `
  --books-dir books `
  --out-dir checkpoints `
  --max-iters 3000
```

## Chat With the Model

After training, start the interactive chat script:

```bash
python chat_nano_llm.py
```

You should see output similar to:

```text
Loaded checkpoint: checkpoints/nano_llm.pt
Using device: cuda
Block size: 128
Type a prompt and press Enter. Type 'exit' to quit.
```

Then enter a prompt:

```text
You: Alice was beginning to get very tired
Model: ...
```

Type `exit` or `quit` to stop.

### Chat Options

```bash
python chat_nano_llm.py --max-new-tokens 500
```

Use a specific checkpoint:

```bash
python chat_nano_llm.py \
  --checkpoint checkpoints/nano_llm.pt \
  --tokenizer checkpoints/tokenizer.json
```

Force CPU:

```bash
python chat_nano_llm.py --device cpu
```

## Important Notes

- The tokenizer is trained from the text extracted from the PDFs.
- The checkpoint and tokenizer must have the same vocabulary size.
- Better output usually requires more data, more training steps, or a larger model.
- Training on CPU can be slow.
- The saved checkpoint includes the model architecture settings used during training.

## Main Scripts

### `train_nano_llm.py`

Contains:

- PDF text loading
- Text cleaning
- `BPETokenizer`
- Transformer model classes
- Training loop
- Loss evaluation
- Checkpoint saving
- Sample generation

### `chat_nano_llm.py`

Contains:

- Checkpoint loading
- Tokenizer loading
- Prompt encoding
- Interactive terminal chat
- Text generation

## Model Architecture

The core model is `NanoGPT`, a small decoder-only transformer:

```text
Input token IDs
    |
    v
Token embeddings + positional embeddings
    |
    v
Transformer blocks
    |
    v
Final layer normalization
    |
    v
Language model head
    |
    v
Next-token probabilities
```

Each transformer block includes causal multi-head self-attention and a feed-forward network with residual connections.

## Example Workflow

```bash
# 1. Install dependencies
pip install torch pypdf tokenizers

# 2. Add PDF files to books/

# 3. Train the tokenizer
python train_tokenizerr.py

# 4. Train the model
python train_nano_llm.py

# 5. Chat with the saved checkpoint
python chat_nano_llm.py
```

## Limitations

This is a learning project, not a production LLM. Because it uses a small transformer trained on a limited dataset, it may produce repetitive, misspelled, or nonsensical text. That is expected for a tiny model.

## License

Add your chosen license here, such as MIT, Apache-2.0, or another license that fits your project.
