from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from train_nano_llm import load_pdf_text

text = load_pdf_text("books")

Path("dataset.txt").write_text(text, encoding="utf8")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

tokenizer.train(["dataset.txt"], trainer)
tokenizer.save("tokenizer.json")

print("Tokenizer trained.")
