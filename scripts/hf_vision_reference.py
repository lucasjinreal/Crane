#!/usr/bin/env python3
"""
HuggingFace Gemma4 Vision Encoder — capture intermediate values via hooks.
Usage: python3 scripts/hf_vision_reference.py
"""

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

MODEL_PATH = "/home/emre/models/gemma-4-E2B-it/"
IMAGE_PATH = "/home/emre/models/Red_Apple.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

def pt(name, t):
    f = t.float()
    try:
        if t.dim() >= 4: v = f[0,0,0,:5].tolist()
        elif t.dim() >= 3: v = f[0,0,:5].tolist()
        elif t.dim() == 2: v = f[0,:5].tolist()
        else: v = f[:5].tolist()
    except: v = f.flatten()[:5].tolist()
    print(f"[HF] {name}")
    print(f"[HF]   shape={tuple(t.shape)} norm={f.norm().item():.4f} min={f.min().item():.6f} max={f.max().item():.6f}")
    print(f"[HF]   first5={[round(x,6) for x in v]}")

print(f"[HF] Device={DEVICE}")
print("[HF] Loading model...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"[HF] Image size (WxH): {image.size}")

conv = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(conv, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt").to(DEVICE)

pv = inputs["pixel_values"].to(DTYPE)
pt("pixel_values", pv)
n_img = (inputs["input_ids"] == 258880).sum().item()
print(f"[HF] input_ids shape={inputs['input_ids'].shape} n_img_tokens={n_img}")

# ── Register hooks ──
captured = {}
inp_captured = {}
hooks = []

def make_hook(key):
    def fn(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        if isinstance(t, torch.Tensor):
            captured[key] = t.detach().cpu()
    return fn

def make_input_hook(key):
    def fn(m, i, o):
        if isinstance(i, tuple) and len(i) > 0 and isinstance(i[0], torch.Tensor):
            inp_captured[key] = i[0].detach().cpu()
    return fn

vt = model.model.vision_tower
pe = vt.patch_embedder
enc = vt.encoder
pool = vt.pooler
ev = model.model.embed_vision

# Patch embedder: capture input (pixel_values after normalization) and output
hooks.append(pe.register_forward_hook(make_hook("patch_embedder")))
hooks.append(pe.input_proj.register_forward_hook(make_hook("input_proj")))
hooks.append(pe.input_proj.register_forward_hook(make_input_hook("input_proj_input")))

# Encoder rotary embedding
hooks.append(enc.rotary_emb.register_forward_hook(make_hook("rotary_emb")))

# Layer 0 details
L0 = enc.layers[0]
hooks.append(L0.input_layernorm.register_forward_hook(make_hook("L0_input_ln")))
hooks.append(L0.self_attn.q_proj.register_forward_hook(make_hook("L0_q_proj")))
hooks.append(L0.self_attn.k_proj.register_forward_hook(make_hook("L0_k_proj")))
hooks.append(L0.self_attn.v_proj.register_forward_hook(make_hook("L0_v_proj")))
if hasattr(L0.self_attn, 'q_norm'):
    hooks.append(L0.self_attn.q_norm.register_forward_hook(make_hook("L0_q_norm")))
if hasattr(L0.self_attn, 'k_norm'):
    hooks.append(L0.self_attn.k_norm.register_forward_hook(make_hook("L0_k_norm")))
hooks.append(L0.self_attn.o_proj.register_forward_hook(make_hook("L0_o_proj")))
hooks.append(L0.self_attn.register_forward_hook(make_hook("L0_self_attn")))
hooks.append(L0.post_attention_layernorm.register_forward_hook(make_hook("L0_post_attn_ln")))
hooks.append(L0.pre_feedforward_layernorm.register_forward_hook(make_hook("L0_pre_ff_ln")))
hooks.append(L0.mlp.register_forward_hook(make_hook("L0_mlp")))
hooks.append(L0.post_feedforward_layernorm.register_forward_hook(make_hook("L0_post_ff_ln")))
hooks.append(L0.register_forward_hook(make_hook("L0_full")))

# Last layer
hooks.append(enc.layers[-1].register_forward_hook(make_hook("L15_full")))

# Pooler
hooks.append(pool.register_forward_hook(make_hook("pooler")))

# Vision tower
hooks.append(vt.register_forward_hook(make_hook("vision_tower")))

# embed_vision
hooks.append(ev.register_forward_hook(make_hook("embed_vision")))
if hasattr(ev, 'embedding_pre_projection_norm'):
    hooks.append(ev.embedding_pre_projection_norm.register_forward_hook(make_hook("embed_vision_norm")))
hooks.append(ev.embedding_projection.register_forward_hook(make_hook("embed_vision_proj")))

# Also capture the pixel_position_ids fed to patch_embedder
# by hooking the vision model's forward to capture its inputs
def vision_tower_input_hook(m, i, o):
    # The vision tower forward takes: pixel_values, pixel_position_ids, padding_positions
    if len(i) >= 3:
        inp_captured["vt_pixel_values"] = i[0].detach().cpu()
        inp_captured["vt_pixel_position_ids"] = i[1].detach().cpu()
        inp_captured["vt_padding_positions"] = i[2].detach().cpu()
    elif len(i) >= 1:
        inp_captured["vt_input_0"] = i[0].detach().cpu()
hooks.append(vt.register_forward_hook(vision_tower_input_hook))

# ── Run forward ──
print("\n[HF] Running forward pass...")
with torch.no_grad():
    outputs = model(**inputs, return_dict=True)
print("[HF] Forward done.\n")

for h in hooks:
    h.remove()

# ── Print results ──
print("[HF] ════════════════════════════════════════")
print("[HF]  INPUT TENSORS TO VISION TOWER")
print("[HF] ════════════════════════════════════════")
for key in sorted(inp_captured.keys()):
    t = inp_captured[key]
    if t.dim() <= 2:
        print(f"[HF] {key}: shape={tuple(t.shape)} dtype={t.dtype}")
        if "position" in key:
            print(f"[HF]   first_rows={t[0,:5].tolist()}")
        if "padding" in key:
            print(f"[HF]   sum={t.sum().item()}")
    else:
        pt(key, t)

print("\n[HF] ════════════════════════════════════════")
print("[HF]  INTERMEDIATE VALUES")
print("[HF] ════════════════════════════════════════\n")

order = [
    "input_proj", "patch_embedder",
    "L0_input_ln", "L0_q_proj", "L0_k_proj", "L0_v_proj",
    "L0_q_norm", "L0_k_norm",
    "L0_o_proj", "L0_self_attn",
    "L0_post_attn_ln", "L0_pre_ff_ln", "L0_mlp", "L0_post_ff_ln",
    "L0_full",
    "L15_full",
    "pooler", "vision_tower",
    "embed_vision_norm", "embed_vision_proj", "embed_vision",
    "rotary_emb",
]

for key in order:
    if key in captured:
        pt(key, captured[key])
    else:
        print(f"[HF] {key}: NOT CAPTURED")

# Extra: RoPE details
if "rotary_emb" in captured:
    t = captured["rotary_emb"]
    print(f"[HF] rotary_emb output type: {type(t)}")
    if isinstance(t, torch.Tensor):
        pt("rotary_emb tensor", t)

# Also print inv_freq
if hasattr(enc.rotary_emb, 'inv_freq'):
    inv = enc.rotary_emb.inv_freq
    print(f"[HF] rotary inv_freq: shape={inv.shape} first5={inv.float()[:5].tolist()}")

# Summary
print(f"\n[HF] All captured keys: {sorted(captured.keys())}")
print(f"[HF] All input keys: {sorted(inp_captured.keys())}")
print("[HF] Done.")
