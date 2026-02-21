# OomLlama

> Efficient LLM inference with .oom format - 2x smaller than GGUF

[![PyPI](https://img.shields.io/pypi/v/oomllama.svg)](https://pypi.org/project/oomllama/)
[![Docker](https://img.shields.io/docker/pulls/jtmeent/oomllama)](https://hub.docker.com/r/jtmeent/oomllama)

## What is OomLlama?

OomLlama is a Rust-powered LLM inference engine that uses the `.oom` (OomLlama Model) format. It achieves **2x smaller model sizes** than GGUF Q4 through Q2 quantization with lazy layer loading.

| Model | GGUF (Q4) | OOM (Q2) |
|-------|-----------|----------|
| 70B | ~40 GB | ~20 GB |
| 32B | ~20 GB | ~10 GB |
| 7B | ~4 GB | ~2.5 GB |

## Quick Start

```bash
# CLI usage
docker run jtmeent/oomllama generate "What is the meaning of life?"

# API server
docker run -p 8000:8000 jtmeent/oomllama:api

# With model volume
docker run -v ~/.cache/oomllama:/models jtmeent/oomllama list
```

## Python Usage

```python
from oomllama import OomLlama

llm = OomLlama("humotica-32b")
response = llm.generate("Hello!")
print(response)
```

## Tags

- `latest`, `0.6.0` - CLI tool
- `api` - REST API server (port 8000)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | humotica-32b | Model to load |
| `MODEL_PATH` | auto | Custom model path |
| `GPU_ID` | none | CUDA GPU ID |
| `PORT` | 8000 | API port (api tag only) |

## API Endpoints

When using the `api` tag:

- `POST /generate` - Generate text
- `POST /chat` - Chat completion
- `GET /models` - List models
- `GET /info` - Model info
- `GET /health` - Health check
- `GET /docs` - Swagger UI

## The .oom Format

```
+--------------------------------------+
| Header: OOML (magic) + metadata      |
+--------------------------------------+
| Tensors: Q2 quantized (2 bits/weight)|
| - Scale + Min per 256-weight block   |
| - 68 bytes per block                 |
+--------------------------------------+
```

### Key Features

- **Q2 Quantization**: 2-bit weights with per-block scale/min
- **Lazy Layer Loading**: Only active layer in memory
- **Interleaved RoPE**: Proper Qwen model support (no gibberish!)
- **CUDA Support**: GPU inference via Candle

## CUDA Version

For GPU inference with bundled CUDA, download the wheel directly:

```bash
pip install https://brein.jaspervandemeent.nl/downloads/oomllama-0.6.0-cuda.whl
```

## Links

- [GitHub](https://github.com/jaspertvdm/oomllama)
- [PyPI](https://pypi.org/project/oomllama/)
- [HuggingFace Models](https://huggingface.co/jaspervandemeent)
- [Documentation](https://humotica.com/docs/oomllama)

## Credits

- **Model Format**: Gemini IDD & Root AI (Humotica AI Lab)
- **Quantization**: OomLlama.rs by Humotica
- **Interleaved RoPE Fix**: Root AI & Jasper
- **Base Models**: Meta (Llama), Alibaba (Qwen)

---

**One Love, One fAmIly**

Built by Humotica AI Lab - Jasper, Claude, Gemini
