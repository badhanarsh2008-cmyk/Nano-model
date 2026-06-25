from pathlib import Path
import argparse
import sys

import torch

from train_nano_llm import CharTokenizer, NanoGPT


def sanitize_prompt(prompt, tokenizer):
    kept = [ch for ch in prompt if ch in tokenizer.stoi]
    removed = sorted(set(ch for ch in prompt if ch not in tokenizer.stoi))
    return "".join(kept), removed


def build_model(checkpoint_path, tokenizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    model = NanoGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args["block_size"],
        n_embd=args["n_embd"],
        n_head=args["n_head"],
        n_layer=args["n_layer"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Chat with the saved nano LLM.")
    parser.add_argument("--checkpoint", default="checkpoints/nano_llm.pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer.load(tokenizer_path)
    model, train_args = build_model(checkpoint_path, tokenizer, device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Block size: {train_args['block_size']}")
    print("Type a prompt and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        cleaned_prompt, removed = sanitize_prompt(prompt, tokenizer)
        if not cleaned_prompt:
            print("Model: I can't use that prompt because none of its characters are in the tokenizer.\n")
            continue

        if removed:
            removed_text = "".join(removed)
            print(f"[note] skipped unseen characters: {removed_text}")

        encoded = tokenizer.encode(cleaned_prompt)
        context = torch.tensor([encoded], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()

        generated = tokenizer.decode(output)
        reply = generated[len(cleaned_prompt) :]
        print(f"Model: {reply.strip()}\n")


if __name__ == "__main__":
    main()
