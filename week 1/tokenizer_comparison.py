import os
# Suppress unnecessary Hugging Face logging warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

# Sample test corpus containing technical terms and acronyms
corpus = "Generative AI and Large Language Models (LLMs) utilize tokenization to process text efficiently."

print("Loading tokenizers from Hugging Face Hub...\n")

# 1. Initialize Byte-Pair Encoding (BPE) Tokenizer
tokenizer_bpe = AutoTokenizer.from_pretrained("gpt2")

# 2. Initialize WordPiece Tokenizer
tokenizer_wp = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3. Initialize SentencePiece Tokenizer
tokenizer_sp = AutoTokenizer.from_pretrained("t5-small")

def evaluate_tokenizer(name, tokenizer, text):
    # Convert raw text into subword token strings
    tokens = tokenizer.tokenize(text)
    # Convert raw text into integer token IDs
    token_ids = tokenizer.encode(text)
    
    print(f"==================================================")
    print(f" {name}")
    print(f"==================================================")
    print(f"Total Token Count : {len(tokens)}")
    print(f"Tokens Breakdown  : {tokens}")
    print(f"Token IDs         : {token_ids[:10]}... (Showing first 10 IDs)")
    print(f"Vocabulary Size   : {tokenizer.vocab_size}\n")

# Run evaluation for each tokenizer strategy
evaluate_tokenizer("Byte-Pair Encoding (GPT-2)", tokenizer_bpe, corpus)
evaluate_tokenizer("WordPiece (BERT Uncased)", tokenizer_wp, corpus)
evaluate_tokenizer("SentencePiece (T5 Small)", tokenizer_sp, corpus)