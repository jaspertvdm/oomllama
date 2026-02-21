# OomLlama

> Efficient LLM inference with .oom format - 2x smaller than GGUF

[![PyPI](https://img.shields.io/pypi/v/oomllama.svg)](https://pypi.org/project/oomllama/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-orange)](https://huggingface.co/jaspervandemeent)

## Quick Start

```python
from oomllama import OomLlama

llm = OomLlama("humotica-7b")
response = llm.generate("What is the meaning of life?")
print(response)
```

## Why OomLlama?

| Feature | GGUF (Q4) | OOM (Q2) |
|---------|-----------|----------|
| 70B Model Size | ~40 GB | ~20 GB |
| 32B Model Size | ~20 GB | ~10 GB |
| RAM Usage | High | Lazy Loading |
| Format | Open | Open (MIT) |

OomLlama uses Q2 quantization with lazy layer loading to run large models on consumer hardware.

## Installation

```bash
pip install oomllama
```

## Features

- **Q2 Quantization**: 2-bit weights with per-block scale/min
- **Lazy Layer Loading**: Only active layer in memory
- **Interleaved RoPE**: Proper Qwen model support (no gibberish!)
- **HuggingFace Integration**: Download models directly
- **GPU Inference**: CUDA support via Candle

## Available Models

| Model | Parameters | Size (.oom) | HuggingFace |
|-------|------------|-------------|-------------|
| humotica-7b | 7B | ~2.5 GB | [Link](https://huggingface.co/jaspervandemeent/humotica-7b-oom) |
| humotica-32b | 32B | ~10 GB | [Link](https://huggingface.co/jaspervandemeent/humotica-32b) |
| LlamaOhm-70B | 70B | ~20 GB | [Link](https://huggingface.co/jaspervandemeent/LlamaOhm-70B) |

## The .oom Format

OOM (OomLlama Model) is a compact model format:

```
+--------------------------------------+
| Header: OOML (magic) + metadata      |
+--------------------------------------+
| Tensors: Q2 quantized (2 bits/weight)|
| - Scale + Min per 256-weight block   |
| - 68 bytes per block                 |
+--------------------------------------+
```

## CLI Usage

```bash
# Run inference
oomllama generate "Hello, world!"

# Convert GGUF to OOM
gguf2oom model.gguf model.oom --quant q2

# Check model info
oomllama info model.oom
```

## Technical Details

### Q2 Quantization

Each weight is stored as 2 bits (0, 1, 2, or 3) with per-block scale and minimum:

```
weight = q2_value * scale + min
```

### Interleaved RoPE (Qwen Fix)

OomLlama supports both LLaMA-style and Qwen-style RoPE:

- **LLaMA-style**: Split at half_dim `[x0:half, x1:half]`
- **Qwen-style (interleaved)**: Even/odd pairs `[x0, x1, x0, x1, ...]`

This fix prevents the "Chinese characters / gibberish" issue with Qwen models.

### Lazy Layer Loading

```
Forward Pass:
  Layer 0: Load -> Compute -> Unload
  Layer 1: Load -> Compute -> Unload
  ...
  Layer N: Load -> Compute -> Unload
```

This enables running 70B models on 24GB GPU RAM.

## Credits

- **Model Format**: Gemini IDD & Root AI (Humotica AI Lab)
- **Quantization**: OomLlama.rs by Humotica
- **Interleaved RoPE Fix**: Root AI & Jasper
- **Base Models**: Meta (Llama), Alibaba (Qwen)

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
