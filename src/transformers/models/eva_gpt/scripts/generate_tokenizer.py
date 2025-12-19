import os
import glob
import json
import sentencepiece as spm
import pandas as pd
from transformers import LlamaTokenizer, LlamaTokenizerFast

# -----------------------------
# 1) Trainiere deinen SentencePiece-Tokenizer (BPE)
# -----------------------------
model_prefix = "tokenizer"
model_type = "bpe"  # WICHTIG: klein schreiben für SentencePiece
base_vocab_size = 35300

# Sammle alle Textdateien
text_dir = "./Text"
text_files = sorted(glob.glob(os.path.join(text_dir, "train_data.txt")))
if not text_files:
    raise FileNotFoundError(f"Keine Trainings-Textdateien in '{text_dir}' gefunden.")
input_arg = ",".join(text_files)

# Optional: Tokens/Phrasen aus CSV, die im SentencePiece-Vocab landen sollen,
# aber NICHT als HuggingFace-Spezialtokens behandelt werden sollen.
special_phrases_csv = "./Text/special_phrases.csv"
custom_tokens = []
if os.path.exists(special_phrases_csv):
    df_phrases = pd.read_csv(special_phrases_csv)

    # WICHTIG:
    # - SentencePiece nutzt "▁" (U+2581) als Whitespace-Markierung.
    # - Wir erzwingen Multi-Word-Phrasen als ein Piece, indem wir Spaces zu ▁ machen.
    # - Optional: Falls deine Phrasen auch am Wortanfang matchen sollen, kann ein führendes ▁ sinnvoll sein.
    df_phrases["token"] = (
        df_phrases["text"].astype(str)
        .str.strip()
        .str.replace(" ", "▁", regex=False)
    )

    custom_tokens = df_phrases["token"].dropna().unique().tolist()
    print(f"✅ {len(custom_tokens)} Tokens/Phrasen aus CSV verarbeitet (als normale Tokens).")
else:
    print(f"⚠️ CSV-Datei '{special_phrases_csv}' nicht gefunden; keine Custom-Tokens.")

# Vocab-Size: Basis + erzwungene Pieces aus CSV
vocab_size = base_vocab_size + len(custom_tokens)

# Trainiere das SentencePiece-BPE-Modell
# WICHTIG:
# - user_defined_symbols: sorgt dafür, dass diese Pieces im Vocab vorhanden sind
# - aber wir behandeln sie später NICHT als HuggingFace special tokens.
spm.SentencePieceTrainer.train(
    input=input_arg,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,                 # "bpe"
    user_defined_symbols=custom_tokens,    # NUR CSV-Phrasen (normale Tokens)
    character_coverage=1.0,
    max_sentence_length=1073741824,
    max_sentencepiece_length=24,
    split_digits=False,
    allow_whitespace_only_pieces=True,
    byte_fallback=True,                    # BPE + byte_fallback
    train_extremely_large_corpus=True,
    input_sentence_size=10000000,
    shuffle_input_sentence=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    pad_piece="<pad>",
)
print(f"🎉 SentencePiece-BPE-Model '{model_prefix}.model' erzeugt.")

# -----------------------------
# 2) Funktionen für Config-Dateien
#    (CSV-Tokens NICHT als special eintragen!)
# -----------------------------
def create_tokenizer_config(
    out_path="tokenizer_config.json",
):
    config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "add_prefix_space": None,

        # Nur echte Specials hier eintragen
        "added_tokens_decoder": {
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "3": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },

        # WICHTIG: leer lassen, damit CSV-Tokens NICHT special sind
        "additional_special_tokens": [],

        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "cls_token": "</s>",
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": "<pad>",
        "sep_token": "</s>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
    }
    with open(out_path, "w", encoding="utf-8") as json_file:
        json.dump(config, json_file, indent=2, ensure_ascii=False)

def create_special_tokens_map(out_path="special_tokens_map.json"):
    special_map = {
        "bos_token": {
            "content": "<s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "eos_token": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "sep_token": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "unk_token": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
        "pad_token": {
            "content": "<pad>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
        },
    }
    with open(out_path, "w", encoding="utf-8") as json_file:
        json.dump(special_map, json_file, indent=2, ensure_ascii=False)

# Erstelle die Config-Dateien
create_tokenizer_config("tokenizer_config.json")
create_special_tokens_map("special_tokens_map.json")
print("✅ tokenizer_config.json und special_tokens_map.json erstellt.")

# -----------------------------
# 3) Lege ein Tokenizer-Verzeichnis (slow) an
# -----------------------------
tokenizer_dir = "tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

for fname in (
    f"{model_prefix}.model",
    f"{model_prefix}.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
):
    os.replace(fname, os.path.join(tokenizer_dir, fname))

print("✅ Tokenizer-Verzeichnis 'tokenizer/' angelegt.")

# Test, ob der slow-Tokenizer lädt
slow_tok = LlamaTokenizer.from_pretrained(tokenizer_dir)

# WICHTIG: NICHT add_special_tokens() auf custom_tokens,
# sonst werden sie als Special Tokens markiert!
print("✅ LlamaTokenizer (slow) erfolgreich geladen.")

# -----------------------------
# 4) Erzeuge einen Fast-Tokenizer in separatem Verzeichnis
# -----------------------------
fast_tokenizer_dir = "tokenizer_fast"
os.makedirs(fast_tokenizer_dir, exist_ok=True)

fast_tok = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
fast_tok.save_pretrained(fast_tokenizer_dir)

print("✅ Fast-Tokenizer-Verzeichnis 'tokenizer_fast/' mit tokenizer.json angelegt.")

# -----------------------------
# 5) Optional: Validierungs-Checks
# -----------------------------
if custom_tokens:
    sample = custom_tokens[:10]
    ids_slow = [slow_tok.convert_tokens_to_ids(t) for t in sample]
    specials_slow = [t in slow_tok.all_special_tokens for t in sample]

    ids_fast = [fast_tok.convert_tokens_to_ids(t) for t in sample]
    specials_fast = [t in fast_tok.all_special_tokens for t in sample]

    print("\n--- CHECK (slow) ---")
    for t, i, s in zip(sample, ids_slow, specials_slow):
        print(f"{t!r} -> id={i}  special={s}")

    print("\n--- CHECK (fast) ---")
    for t, i, s in zip(sample, ids_fast, specials_fast):
        print(f"{t!r} -> id={i}  special={s}")
else:
    print("ℹ️ Keine CSV-Tokens zum Prüfen vorhanden.")
