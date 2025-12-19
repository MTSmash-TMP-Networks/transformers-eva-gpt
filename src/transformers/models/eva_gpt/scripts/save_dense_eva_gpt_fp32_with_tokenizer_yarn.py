# save_dense_eva_gpt_fp32_with_tokenizer_yarn.py
import os, shutil, torch, json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

# -----------------------------
# Pfade
# -----------------------------
TOKENIZER_SRC = "./tokenizer_fast/"
OUT_DIR = "./eva-mini2048-eva_gpt-dense-fp32"

ORIG_MAX_POS = 4096
NEW_MAX_POS = 8192
ROPE_FACTOR = NEW_MAX_POS / ORIG_MAX_POS  # 2.0

# Zielordner frisch anlegen
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Tokenizer laden (SLOW = sentencepiece)
# -----------------------------
tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC, use_fast=False)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token or tok.bos_token

vocab_size = len(tok)
bos_id = tok.bos_token_id
eos_id = tok.eos_token_id
pad_id = tok.pad_token_id

print(f"[Tokenizer] vocab={vocab_size}, bos={bos_id}, eos={eos_id}, pad={pad_id}")

# -----------------------------
# 2) EvaGPT Config erzeugen (WICHTIG!)
# -----------------------------
cfg = AutoConfig.for_model("eva_gpt")

# --- Core dims ---
cfg.vocab_size = vocab_size
cfg.hidden_size = 1792
cfg.intermediate_size = 3584
cfg.num_hidden_layers = 18
cfg.num_attention_heads = 14
cfg.num_key_value_heads = 14
cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads  # 128

# --- Attention / Norm ---
cfg.hidden_act = "silu"
cfg.attention_bias = True
cfg.attention_dropout = 0.0
cfg.rms_norm_eps = 1e-5
cfg.initializer_range = 0.02

# --- Positional ---
cfg.max_position_embeddings = NEW_MAX_POS
cfg.sliding_window = 4096

cfg.layer_types = [
    "sliding_attention" if i % 2 == 0 else "full_attention"
    for i in range(cfg.num_hidden_layers)
]

# --- RoPE (YaRN, korrekt für EvaGPT) ---
cfg.rope_parameters = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": ROPE_FACTOR,
    "truncate": False,
    "original_max_position_embeddings": ORIG_MAX_POS,
}

# --- Tokens ---
cfg.bos_token_id = bos_id
cfg.eos_token_id = eos_id
cfg.pad_token_id = pad_id

# --- MoE: AUS (dense) ---
cfg.num_local_experts = 1
cfg.num_experts_per_tok = 1
cfg.router_aux_loss_coef = 0.0
cfg.output_router_logits = False

# --- Sonstiges ---
cfg.tie_word_embeddings = False
cfg.use_cache = True
cfg._attn_implementation = "eager"

# -----------------------------
# 3) Modell bauen
# -----------------------------
model = AutoModelForCausalLM.from_config(cfg)

# Sanity-Check: MoE wirklich aus
exp = model.model.layers[0].mlp.experts
print(
    "[Sanity]",
    "num_experts =", getattr(exp, "num_experts", None),
    "gate_up_proj =", tuple(exp.gate_up_proj.shape),
)

# -----------------------------
# 4) FP32 + CPU
# -----------------------------
model.to(dtype=torch.float32, device="cpu")
print("[Model] dtype:", next(model.parameters()).dtype)

# -----------------------------
# 5) Single-file safetensors
# -----------------------------
state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
save_file(state, os.path.join(OUT_DIR, "model.safetensors"))

# -----------------------------
# 6) Config + Tokenizer speichern
# -----------------------------
cfg.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# Falls HF Dateien nicht kopiert hat
for fname in [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "added_tokens.json",
]:
    src = os.path.join(TOKENIZER_SRC, fname)
    dst = os.path.join(OUT_DIR, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

# -----------------------------
# 7) Parameter-Check
# -----------------------------
params = sum(p.numel() for p in model.parameters())
print(f"[Model] Params: {params:,}")
print(f"[Model] Size ~ {params * 4 / 1e9:.2f} GB (fp32)")
print("[Done] Saved to:", OUT_DIR)
print("Files:", sorted(os.listdir(OUT_DIR)))

