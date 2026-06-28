# Nano LLM Architecture

Step 1: Collect PDFs

Step 2: Extract text from each PDF

Step 3: Clean common PDF encoding artifacts

Step 4: Train or load the BPE tokenizer

Step 5: Encode text into token IDs

Step 6: Create input-target pairs

Step 7: Train the model

```text
Token IDs
  -> Token embeddings + positional embeddings
  -> Causal self-attention
  -> Multi-head attention
  -> Transformer blocks
  -> Language model head
  -> Next-token logits
```

## Current Pipeline

```text
Raw PDFs
  -> Clean text
  -> BPE tokenizer
  -> Token IDs
  -> Embeddings
  -> Transformer blocks
  -> Cross entropy loss
  -> AdamW optimizer
  -> Checkpoint saving
  -> Chat interface
```
