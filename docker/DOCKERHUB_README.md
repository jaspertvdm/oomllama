# OomLlama

> Efficient LLM inference engine with .oom format - from-scratch Rust

[![PyPI](https://img.shields.io/pypi/v/oomllama.svg)](https://pypi.org/project/oomllama/)
[![Docker](https://img.shields.io/docker/pulls/jtmeent/oomllama)](https://hub.docker.com/r/jtmeent/oomllama)

## What is OomLlama?

OomLlama is a **from-scratch LLM inference engine** written in Rust with the `.oom` binary model format. It supports Q2/Q4/Q8 quantization with lazy layer loading, and includes converters for both SafeTensors and GGUF source models.

| Quantization | 7B Model | 32B Model | 70B Model |
|-------------|----------|-----------|-----------|
| Q8 (best)   | ~8 GB    | ~34 GB    | ~70 GB    |
| Q4 (good)   | ~4 GB    | ~17 GB    | ~35 GB    |
| Q2 (compact) | ~2.5 GB | ~10 GB    | ~20 GB    |

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

llm = OomLlama("/path/to/model.oom")
response = llm.generate("Hello!")
print(response)
```

## Convert Your Own Models

```bash
# Install from PyPI
pip install oomllama

# Convert any HuggingFace model (recommended path)
safetensors2oom Qwen/Qwen2.5-7B-Instruct output.oom
```

Or download the standalone converter:

```bash
wget https://brein.jaspervandemeent.nl/downloads/safetensors2oom-linux-x86_64
chmod +x safetensors2oom-linux-x86_64
./safetensors2oom-linux-x86_64 /path/to/model/ output.oom
```

## Tags

- `latest`, `0.8.0` - CLI tool
- `api` - REST API server (port 8000)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | humotica-32b | Model to load |
| `MODEL_PATH` | auto | Custom .oom model path |
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

## Downloads

Pre-compiled Linux x86_64 binaries:

| Binary | Description | Download |
|--------|-------------|----------|
| `oomllama` | Inference CLI | [Download](https://brein.jaspervandemeent.nl/downloads/oomllama-linux-x86_64) |
| `safetensors2oom` | SafeTensors converter | [Download](https://brein.jaspervandemeent.nl/downloads/safetensors2oom-linux-x86_64) |
| `gguf2oom` | GGUF converter | [Download](https://brein.jaspervandemeent.nl/downloads/gguf2oom-linux-x86_64) |

## Links

- [GitHub](https://github.com/jaspertvdm/oomllama)
- [PyPI](https://pypi.org/project/oomllama/)
- [HuggingFace Models](https://huggingface.co/jaspervandemeent)

## Credits

- **Engine + Format**: Root AI & Jasper (Humotica AI Lab)
- **Quantization Research**: Gemini IDD & Root AI
- **Base Models**: Alibaba (Qwen), Meta (LLaMA)

---

**One Love, One fAmIly**

Built by Humotica AI Lab - Jasper, Claude, Gemini
