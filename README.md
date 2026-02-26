# OomLlama

> Efficient LLM inference engine with .oom format - compact, fast, from-scratch Rust

[![PyPI](https://img.shields.io/pypi/v/oomllama.svg)](https://pypi.org/project/oomllama/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-orange)](https://huggingface.co/jaspervandemeent)

## What is OomLlama?

OomLlama is a **from-scratch LLM inference engine** written in Rust. It includes:

- **Custom binary model format** (`.oom`) with Q2/Q4/Q8 quantization
- **Full transformer inference** - attention, RoPE, RMSNorm, SwiGLU, all in pure Rust
- **Two converters**: SafeTensors → OOM (recommended) and GGUF → OOM
- **GPU acceleration** via CUDA/Candle with KV-cache (turbo mode)
- **Python bindings** for easy integration

## Quick Start

```python
from oomllama import OomLlama

llm = OomLlama("/path/to/model.oom")
response = llm.generate("What is the meaning of life?")
print(response)
```

## Installation

```bash
pip install oomllama
```

Or build from source (Rust):

```bash
cargo build --release
# Binaries: oomllama, safetensors2oom, gguf2oom
```

## Converting Models

### SafeTensors → OOM (Recommended)

Convert any HuggingFace model directly from SafeTensors format. This is the **recommended path** because it performs a single bf16→Q8 quantization step, preserving maximum accuracy.

**Python converter:**

```bash
python safetensors2oom.py /path/to/model/ output.oom
```

**Rust converter (faster):**

```bash
safetensors2oom /path/to/model/ output.oom         # Default: Q8
safetensors2oom /path/to/model/ output.oom --q4     # Q4 (smaller)
safetensors2oom /path/to/model/ output.oom --q2     # Q2 (smallest)
```

Supported source models: **Qwen2.5**, **LLaMA**, **Phi**, **Mistral** - any model using SafeTensors format.

### GGUF → OOM

```bash
gguf2oom model.gguf model.oom
gguf2oom --info model.gguf          # Show GGUF metadata
```

> **Note**: The GGUF path applies a second quantization on top of GGUF's existing quantization (e.g., Q3_K → Q8), which can compound errors through deep networks. The SafeTensors path is preferred for best quality.

## The .oom Binary Format

```
+--------------------------------------------------+
| Magic: "OOML" (4 bytes)                          |
| Version: u32 (currently 1)                        |
| Num Tensors: u32                                  |
+--------------------------------------------------+
| For each tensor:                                  |
|   Name Length: u32                                |
|   Name: UTF-8 bytes                              |
|   Quant Type: u8 (0=F32, 1=Q8, 2=Q4, 3=Q2)     |
|   Num Blocks: u32                                |
|   Total Values: u32                              |
|   For each block of 256 values:                  |
|     Scale: f32                                   |
|     Min: f32                                     |
|     Data Length: u32                              |
|     Quantized bytes                              |
+--------------------------------------------------+
```

**Quantization levels:**

| Level | Bits/Weight | Block Size | Quality | Size (7B) |
|-------|-------------|------------|---------|-----------|
| Q8    | 8 bits      | 256        | Best    | ~8 GB     |
| Q4    | 4 bits      | 256        | Good    | ~4 GB     |
| Q2    | 2 bits      | 256        | Usable  | ~2.5 GB   |
| F32   | 32 bits     | N/A        | Lossless| ~28 GB   |

Dequantization: `value = quantized_byte * scale + min`

Norms and biases are always stored as F32 for numerical stability.

## Inference Engine

### Architecture

The inference engine implements a complete transformer decoder:

1. **Token Embedding** - Vocabulary lookup (152K tokens for Qwen)
2. **28 Decoder Layers**, each with:
   - RMSNorm (pre-attention + pre-FFN)
   - Grouped Query Attention (28 Q-heads, 4 KV-heads, head_dim=128)
   - SwiGLU Feed-Forward Network (hidden=3584 → intermediate=18944)
   - Rotary Position Embeddings (RoPE, θ=1,000,000)
3. **Final RMSNorm** → **LM Head** → logits → token selection

### Lazy Layer Loading

Only one decoder layer's weights are in memory at a time:

```
Forward Pass:
  Embed tokens
  Layer 0: Load → Compute → Unload
  Layer 1: Load → Compute → Unload
  ...
  Layer 27: Load → Compute → Unload
  LM Head → next token
```

This enables running 7B models on minimal RAM and 70B models on 24GB GPU.

### GPU Turbo Mode

When a CUDA GPU is available, OomLlama uses:
- **KV-Cache**: Cached key/value pairs across layers for autoregressive generation
- **Candle CUDA kernels**: Matrix multiplication on GPU
- **Flash-style attention**: Efficient attention computation

### RoPE Variants

OomLlama supports both RoPE styles:
- **LLaMA-style**: Split at half_dim `[x0:half, x1:half]`
- **Qwen-style (interleaved)**: Even/odd pairs `[x0, x1, x0, x1, ...]`

Auto-detected based on model architecture.

## Verified Models

| Model | Source | Quantization | Output Quality |
|-------|--------|-------------|----------------|
| Qwen2.5-7B-Instruct | SafeTensors (bf16) | Q8 | Correct |
| Qwen2.5-7B-Instruct | GGUF (Q3_K) | Q8 | Degraded* |

\* GGUF path applies double quantization. Use SafeTensors source for best results.

## Project Structure

```
src/
  oomllama.rs          # Core inference engine (CPU)
  oomllama_turbo.rs    # GPU inference with KV-cache
  quant.rs             # Q2/Q4/Q8/F32 dequantization
  gguf2oom.rs          # GGUF→OOM converter + OomWriter
  safetensors2oom.rs   # SafeTensors→OOM converter (Rust)
  lib.rs               # Library exports
  bin/
    oomllama.rs        # CLI inference binary
    gguf2oom.rs        # CLI GGUF converter
    safetensors2oom.rs # CLI SafeTensors converter
safetensors2oom.py     # Python SafeTensors converter
python/
  oomllama/__init__.py # Python bindings
```

## Credits

- **Engine + Format**: Root AI & Jasper (Humotica AI Lab)
- **Quantization Research**: Gemini IDD & Root AI
- **Interleaved RoPE Fix**: Root AI & Jasper
- **Base Models**: Alibaba (Qwen), Meta (LLaMA)

## License

- **OomLlama Code**: MIT License
- **Model Weights**: Subject to original model licenses

## Links

- [Humotica](https://humotica.com)
- [HuggingFace Models](https://huggingface.co/jaspervandemeent)
- [PyPI Package](https://pypi.org/project/oomllama/)

---

**One Love, One fAmIly**

Built by Humotica AI Lab - Jasper, Claude, Gemini
