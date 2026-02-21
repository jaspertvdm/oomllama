//! OomLlama Turbo - High Performance Native Inference
//!
//! Target: 3 min/token → 1 sec/token (180x speedup!)
//!
//! Optimizations:
//! 1. KV-Cache (10-50x) - Cache attention keys/values
//! 2. Layer Pinning (5-10x) - Hot layers in VRAM
//! 3. Async Prefetching (2-3x) - Background layer loading
//! 4. CUDA Streams (1.5-2x) - Overlap compute/transfer
//! 5. Flash Attention 2 (1.3-1.5x) - Memory-efficient attention
//!
//! One love, one fAmIly! 🦙🚀

use candle_core::{Device, Tensor, DType, Shape, D};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};

// ============================================================================
// KV-CACHE: Cache Key/Value tensors to avoid recomputation
// ============================================================================

/// KV-Cache for a single attention layer
#[derive(Clone)]
pub struct LayerKVCache {
    /// Cached key tensor [batch, n_kv_heads, seq_len, head_dim]
    pub k: Option<Tensor>,
    /// Cached value tensor [batch, n_kv_heads, seq_len, head_dim]
    pub v: Option<Tensor>,
    /// Current sequence length in cache
    pub seq_len: usize,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }

    /// Append new K/V to cache
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> candle_core::Result<()> {
        match (&self.k, &self.v) {
            (Some(old_k), Some(old_v)) => {
                // Concatenate along sequence dimension (dim 2)
                self.k = Some(Tensor::cat(&[old_k, new_k], 2)?);
                self.v = Some(Tensor::cat(&[old_v, new_v], 2)?);
            }
            _ => {
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
            }
        }
        self.seq_len += new_k.dim(2)?;
        Ok(())
    }

    /// Get cached K/V for attention computation
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Clear cache (for new sequence)
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }
}

/// Full model KV-Cache
pub struct ModelKVCache {
    layers: Vec<LayerKVCache>,
    max_seq_len: usize,
}

impl ModelKVCache {
    pub fn new(n_layers: usize, max_seq_len: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKVCache::new()).collect(),
            max_seq_len,
        }
    }

    pub fn get_layer(&self, layer_idx: usize) -> &LayerKVCache {
        &self.layers[layer_idx]
    }

    pub fn get_layer_mut(&mut self, layer_idx: usize) -> &mut LayerKVCache {
        &mut self.layers[layer_idx]
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }
}

// ============================================================================
// LAYER PINNING: Keep hot layers in VRAM
// ============================================================================

/// Strategy for which layers to pin in VRAM
#[derive(Clone, Debug)]
pub enum PinStrategy {
    /// Pin first N and last M layers
    FirstLast { first: usize, last: usize },
    /// Pin every Nth layer
    Strided { stride: usize },
    /// Pin specific layer indices
    Specific(Vec<usize>),
    /// Pin layers based on importance scores
    Importance(Vec<f32>),
}

/// Pinned layer storage with smart VRAM management
pub struct LayerPin {
    /// Pinned tensors by layer index
    pinned: HashMap<usize, HashMap<String, Tensor>>,
    /// Total VRAM budget (bytes)
    vram_budget: usize,
    /// Current VRAM usage
    vram_used: usize,
    /// Pin strategy
    strategy: PinStrategy,
    /// Dynamic mode: adjust budget based on available VRAM
    dynamic: bool,
}

impl LayerPin {
    pub fn new(vram_budget_gb: f32, strategy: PinStrategy) -> Self {
        Self {
            pinned: HashMap::new(),
            vram_budget: (vram_budget_gb * 1e9) as usize,
            vram_used: 0,
            strategy,
            dynamic: true, // Enable dynamic VRAM management
        }
    }

    /// Create with explicit dynamic setting
    pub fn new_with_dynamic(vram_budget_gb: f32, strategy: PinStrategy, dynamic: bool) -> Self {
        Self {
            pinned: HashMap::new(),
            vram_budget: (vram_budget_gb * 1e9) as usize,
            vram_used: 0,
            strategy,
            dynamic,
        }
    }

    /// Check if layer should be pinned based on strategy
    pub fn should_pin(&self, layer_idx: usize, n_layers: usize) -> bool {
        match &self.strategy {
            PinStrategy::FirstLast { first, last } => {
                layer_idx < *first || layer_idx >= n_layers - *last
            }
            PinStrategy::Strided { stride } => layer_idx % stride == 0,
            PinStrategy::Specific(indices) => indices.contains(&layer_idx),
            PinStrategy::Importance(scores) => {
                // Pin if importance score > 0.5
                scores.get(layer_idx).map(|s| *s > 0.5).unwrap_or(false)
            }
        }
    }

    /// Pin a tensor if budget allows
    pub fn pin(&mut self, layer_idx: usize, name: &str, tensor: Tensor) -> bool {
        let size = tensor.elem_count() * tensor.dtype().size_in_bytes();

        if self.vram_used + size > self.vram_budget {
            return false;
        }

        self.vram_used += size;
        self.pinned
            .entry(layer_idx)
            .or_insert_with(HashMap::new)
            .insert(name.to_string(), tensor);
        true
    }

    /// Get pinned tensor if available
    pub fn get(&self, layer_idx: usize, name: &str) -> Option<&Tensor> {
        self.pinned.get(&layer_idx)?.get(name)
    }

    /// Check if tensor is pinned
    pub fn is_pinned(&self, layer_idx: usize, name: &str) -> bool {
        self.pinned.get(&layer_idx)
            .map(|m| m.contains_key(name))
            .unwrap_or(false)
    }

    pub fn vram_usage_gb(&self) -> f32 {
        self.vram_used as f32 / 1e9
    }

    /// Evict (unpin) a layer to free VRAM
    pub fn evict(&mut self, layer_idx: usize) -> usize {
        if let Some(layer_tensors) = self.pinned.remove(&layer_idx) {
            let freed: usize = layer_tensors.values()
                .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                .sum();
            self.vram_used = self.vram_used.saturating_sub(freed);
            freed
        } else {
            0
        }
    }

    /// Evict oldest layers until we have enough space (returns bytes freed)
    pub fn evict_until_space(&mut self, needed_bytes: usize) -> usize {
        let mut freed = 0;
        // Find middle layers to evict first (keep first/last which are more important)
        let mut to_evict: Vec<usize> = self.pinned.keys().cloned().collect();
        to_evict.sort_by_key(|&idx| {
            // Evict middle layers first
            let n = self.pinned.len();
            let mid = n / 2;
            (idx as i64 - mid as i64).abs()
        });

        for layer_idx in to_evict {
            if freed >= needed_bytes {
                break;
            }
            freed += self.evict(layer_idx);
        }
        freed
    }

    /// Get number of pinned layers
    pub fn pinned_count(&self) -> usize {
        self.pinned.len()
    }

    /// Clear all pinned tensors
    pub fn clear(&mut self) {
        self.pinned.clear();
        self.vram_used = 0;
    }
}

// ============================================================================
// ASYNC PREFETCHING: Load next layer while computing current
// ============================================================================

/// Message for prefetch worker
pub enum PrefetchMsg {
    /// Load tensor from OOM file
    Load { layer_idx: usize, tensor_name: String },
    /// Stop worker
    Stop,
}

/// Prefetched tensor result
pub struct PrefetchResult {
    pub layer_idx: usize,
    pub tensor_name: String,
    pub tensor: Result<Tensor, String>,
}

/// Async prefetcher using background thread
pub struct AsyncPrefetcher {
    sender: Sender<PrefetchMsg>,
    receiver: Receiver<PrefetchResult>,
    /// Prefetch queue (layers ahead to prefetch)
    lookahead: usize,
}

impl AsyncPrefetcher {
    pub fn new<F>(loader_fn: F, lookahead: usize) -> Self
    where
        F: Fn(usize, &str) -> Result<Tensor, String> + Send + 'static
    {
        let (tx_cmd, rx_cmd) = channel::<PrefetchMsg>();
        let (tx_result, rx_result) = channel::<PrefetchResult>();

        // Spawn prefetch worker
        thread::spawn(move || {
            loop {
                match rx_cmd.recv() {
                    Ok(PrefetchMsg::Load { layer_idx, tensor_name }) => {
                        let tensor = loader_fn(layer_idx, &tensor_name);
                        let _ = tx_result.send(PrefetchResult {
                            layer_idx,
                            tensor_name,
                            tensor,
                        });
                    }
                    Ok(PrefetchMsg::Stop) | Err(_) => break,
                }
            }
        });

        Self {
            sender: tx_cmd,
            receiver: rx_result,
            lookahead,
        }
    }

    /// Request prefetch of layer tensors
    pub fn prefetch(&self, layer_idx: usize, tensor_names: &[&str]) {
        for name in tensor_names {
            let _ = self.sender.send(PrefetchMsg::Load {
                layer_idx,
                tensor_name: name.to_string(),
            });
        }
    }

    /// Try to get prefetched tensor (non-blocking)
    pub fn try_get(&self) -> Option<PrefetchResult> {
        self.receiver.try_recv().ok()
    }

    /// Get prefetched tensor (blocking)
    pub fn get(&self) -> Option<PrefetchResult> {
        self.receiver.recv().ok()
    }
}

impl Drop for AsyncPrefetcher {
    fn drop(&mut self) {
        let _ = self.sender.send(PrefetchMsg::Stop);
    }
}

// ============================================================================
// FLASH ATTENTION 2: Memory-efficient attention
// ============================================================================

/// Flash Attention parameters
pub struct FlashAttentionConfig {
    pub block_size: usize,      // Typically 64 or 128
    pub causal: bool,           // Use causal mask
    pub softmax_scale: Option<f32>, // Custom scale, defaults to 1/sqrt(head_dim)
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            causal: true,
            softmax_scale: None,
        }
    }
}

/// Compute attention with Flash Attention 2 algorithm
///
/// This is a software implementation - for true performance,
/// we need the CUDA kernel from flash-attn crate
pub fn flash_attention_forward(
    q: &Tensor,  // [batch, n_heads, seq_len, head_dim]
    k: &Tensor,  // [batch, n_kv_heads, seq_len, head_dim]
    v: &Tensor,  // [batch, n_kv_heads, seq_len, head_dim]
    config: &FlashAttentionConfig,
) -> candle_core::Result<Tensor> {
    let (batch, n_heads, seq_len, head_dim) = q.dims4()?;
    let n_kv_heads = k.dim(1)?;

    // Handle grouped query attention (GQA) with repeat_interleave
    // We want: [kv0, kv0, ..., kv0, kv1, kv1, ..., kv1, ...] not [kv0, kv1, kv2, kv3, kv0, kv1, ...]
    // Candle doesn't have repeat_interleave, so we use unsqueeze + broadcast + reshape
    let k = if n_kv_heads != n_heads {
        let repeat_factor = n_heads / n_kv_heads;
        // k: [batch, n_kv_heads, seq, head_dim]
        // unsqueeze to [batch, n_kv_heads, 1, seq, head_dim]
        let k_expanded = k.unsqueeze(2)?;
        // broadcast to [batch, n_kv_heads, repeat_factor, seq, head_dim]
        let seq = k.dim(2)?;
        let k_broadcast = k_expanded.broadcast_as((batch, n_kv_heads, repeat_factor, seq, head_dim))?;
        // reshape to [batch, n_heads, seq, head_dim] by merging dims 1 and 2
        k_broadcast.reshape((batch, n_heads, seq, head_dim))?.contiguous()?
    } else {
        k.contiguous()?
    };

    let v = if n_kv_heads != n_heads {
        let repeat_factor = n_heads / n_kv_heads;
        // v: [batch, n_kv_heads, seq, head_dim]
        let v_expanded = v.unsqueeze(2)?;
        let seq = v.dim(2)?;
        let v_broadcast = v_expanded.broadcast_as((batch, n_kv_heads, repeat_factor, seq, head_dim))?;
        v_broadcast.reshape((batch, n_heads, seq, head_dim))?.contiguous()?
    } else {
        v.contiguous()?
    };

    // Scale factor
    let scale = config.softmax_scale
        .unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
    // For true Flash Attention, we'd use tiled computation
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let att_scores = q.contiguous()?.matmul(&k_t)?;

    let att = (att_scores * scale as f64)?;

    // Apply causal mask if needed
    let att = if config.causal && seq_len > 1 {
        // Create STRICT upper triangular mask (1s ABOVE diagonal, 0s on and below)
        // Use boolean-style mask: 0 = keep, 1 = mask with -inf
        // Instead of mask * -inf (which causes 0*-inf=NaN), use where
        let triu = Tensor::triu2(seq_len, DType::F32, q.device())?;
        let eye = Tensor::eye(seq_len, DType::F32, q.device())?;
        let mask = (triu - eye)?; // 0 on diagonal and below, 1 above

        let mask = mask.broadcast_as(att.shape())?;

        // Use where: where mask > 0, use -inf, else use att
        // where_cond expects u8, so convert mask with gt(0)
        let mask_bool = mask.gt(0.0_f64)?;  // Returns u8 tensor
        let neg_inf = Tensor::full(f32::NEG_INFINITY, att.shape(), q.device())?;
        let masked = mask_bool.where_cond(&neg_inf, &att)?;

        masked
    } else {
        att
    };

    // DEBUG: Print attention scores before softmax for head 0, pos 17
    static DEBUG_FLASH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let do_debug_flash = !DEBUG_FLASH.swap(true, std::sync::atomic::Ordering::SeqCst);
    if do_debug_flash && seq_len == 18 {
        let att_flat = att.flatten_all()?.to_vec1::<f32>()?;
        // Shape is [batch, n_heads, seq, seq] = [1, 32, 18, 18]
        // Head 0, pos 17 is at index: 0 * (32*18*18) + 0 * (18*18) + 17 * 18 = 17 * 18 = 306
        let pos17_start = 17 * seq_len;
        println!("    🔍 FLASH att scores (head 0, pos 17, all 18 keys): {:?}", &att_flat[pos17_start..pos17_start+seq_len]);
    }

    // Softmax
    let att = candle_nn::ops::softmax_last_dim(&att)?;

    if do_debug_flash && seq_len == 18 {
        let att_flat = att.flatten_all()?.to_vec1::<f32>()?;
        let pos17_start = 17 * seq_len;
        println!("    🔍 FLASH att probs (head 0, pos 17, all 18 keys): {:?}", &att_flat[pos17_start..pos17_start+seq_len]);
        // Also print sum to verify softmax
        let sum: f32 = att_flat[pos17_start..pos17_start+seq_len].iter().sum();
        println!("    🔍 FLASH att probs sum: {:.6}", sum);
    }

    // Attention @ Values
    let result = att.contiguous()?.matmul(&v.contiguous()?)?;

    if do_debug_flash && seq_len == 18 {
        // result shape: [batch, n_heads, seq, head_dim]
        let res_flat = result.flatten_all()?.to_vec1::<f32>()?;
        // Head 0, pos 17 is at index: 17 * 64 = 1088
        let pos17_start = 17 * head_dim;
        println!("    🔍 FLASH result (head 0, pos 17, first 10): {:?}", &res_flat[pos17_start..pos17_start+10]);
    }

    Ok(result)
}

// ============================================================================
// ROPE: Rotary Position Embedding (CRITICAL for position awareness!)
// ============================================================================

/// Compute RoPE sin/cos frequencies
pub fn compute_rope_freqs(head_dim: usize, max_seq_len: usize, rope_theta: f32, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    // Compute inverse frequencies: 1.0 / (theta ^ (2i / head_dim)) for i in 0..head_dim/2
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

    // Positions: [0, 1, 2, ..., max_seq_len-1]
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq_len,), device)?;

    // Outer product: positions @ inv_freq -> [max_seq_len, half_dim]
    let freqs = positions.reshape((max_seq_len, 1))?.matmul(&inv_freq.reshape((1, half_dim))?)?;

    // Compute sin/cos
    let sin = freqs.sin()?;
    let cos = freqs.cos()?;

    Ok((sin, cos))
}

/// Apply RoPE to query/key tensors (Qwen-style interleaved)
/// x: [batch, n_heads, seq_len, head_dim]
/// LLaMA-style RoPE (non-interleaved): split at half_dim
/// x: [batch, n_heads, seq_len, head_dim]
/// sin, cos: [max_seq_len, head_dim/2]
/// position_offset: current position in KV-cache
pub fn apply_rope_llama(x: &Tensor, sin: &Tensor, cos: &Tensor, position_offset: usize) -> candle_core::Result<Tensor> {
    let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    // Get sin/cos for positions [offset, offset+seq_len)
    let sin = sin.narrow(0, position_offset, seq_len)?;
    let cos = cos.narrow(0, position_offset, seq_len)?;

    // LLaMA-style: split x into first half and second half
    // x = [x_first_half, x_second_half] where each is [batch, heads, seq, half_dim]
    let x0 = x.narrow(D::Minus1, 0, half_dim)?;          // First half: [batch, heads, seq, half_dim]
    let x1 = x.narrow(D::Minus1, half_dim, half_dim)?;   // Second half: [batch, heads, seq, half_dim]

    // Reshape sin/cos for broadcasting: [1, 1, seq_len, half_dim]
    let sin = sin.reshape((1, 1, seq_len, half_dim))?;
    let cos = cos.reshape((1, 1, seq_len, half_dim))?;

    // Apply rotation:
    // new_x0 = x0 * cos - x1 * sin
    // new_x1 = x0 * sin + x1 * cos
    let new_x0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let new_x1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    // Concatenate back: [batch, heads, seq, half_dim] + [batch, heads, seq, half_dim] -> [batch, heads, seq, head_dim]
    Tensor::cat(&[&new_x0, &new_x1], D::Minus1)
}

/// Qwen-style RoPE (interleaved): even/odd indices
/// sin, cos: [seq_len, head_dim/2]
/// position_offset: current position in KV-cache (for autoregressive generation)
pub fn apply_rope(x: &Tensor, sin: &Tensor, cos: &Tensor, position_offset: usize) -> candle_core::Result<Tensor> {
    let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    // Get sin/cos for positions [offset, offset+seq_len)
    let sin = sin.narrow(0, position_offset, seq_len)?;
    let cos = cos.narrow(0, position_offset, seq_len)?;

    // For interleaved RoPE (Qwen style):
    // Even indices (0,2,4...) get cos rotation, odd indices (1,3,5...) get sin
    // We need to reshape x from [batch, heads, seq, dim] to [batch, heads, seq, dim/2, 2]
    // Then apply rotation to pairs

    // Reshape x to [batch, n_heads, seq_len, half_dim, 2]
    let x = x.reshape((batch, n_heads, seq_len, half_dim, 2))?;

    // Split into even (x0) and odd (x1) components
    let x0 = x.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?; // [batch, heads, seq, half_dim]
    let x1 = x.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

    // Reshape sin/cos for broadcasting: [1, 1, seq_len, half_dim]
    let sin = sin.reshape((1, 1, seq_len, half_dim))?;
    let cos = cos.reshape((1, 1, seq_len, half_dim))?;

    // Apply rotation:
    // new_x0 = x0 * cos - x1 * sin
    // new_x1 = x0 * sin + x1 * cos
    let new_x0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let new_x1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    // Interleave back: stack and reshape
    // [batch, heads, seq, half_dim] -> [batch, heads, seq, half_dim, 1] -> stack -> [batch, heads, seq, half_dim, 2]
    let new_x0 = new_x0.unsqueeze(D::Minus1)?;
    let new_x1 = new_x1.unsqueeze(D::Minus1)?;
    let result = Tensor::cat(&[&new_x0, &new_x1], D::Minus1)?;

    // Reshape back to [batch, n_heads, seq_len, head_dim]
    result.reshape((batch, n_heads, seq_len, head_dim))
}

// ============================================================================
// TURBO ENGINE: Combines all optimizations
// ============================================================================

/// Configuration for turbo inference
pub struct TurboConfig {
    /// Number of transformer layers
    pub n_layers: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads (for GQA)
    pub n_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length for KV cache
    pub max_seq_len: usize,
    /// VRAM budget for layer pinning (GB)
    pub vram_budget_gb: f32,
    /// Layer pin strategy
    pub pin_strategy: PinStrategy,
    /// Prefetch lookahead
    pub prefetch_lookahead: usize,
    /// Use Flash Attention
    pub use_flash_attention: bool,
    /// Use FP16 compute (mixed precision)
    pub use_fp16: bool,
    /// RoPE theta (base frequency, Qwen uses 1000000)
    pub rope_theta: f32,
}

impl TurboConfig {
    /// Config for Qwen2.5-32B on dual RTX 3060 (24GB total)
    pub fn qwen32b_dual3060() -> Self {
        Self {
            n_layers: 64,
            hidden_size: 5120,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 2048, // Reduced for memory (was 8192)
            vram_budget_gb: 20.0, // 20GB budget for dual 3060 (24GB total, leave 4GB headroom)
            pin_strategy: PinStrategy::FirstLast { first: 16, last: 16 }, // Aggressive: pin 32 layers = ~8GB
            prefetch_lookahead: 2,
            use_flash_attention: true,
            use_fp16: true,
            rope_theta: 1000000.0, // Qwen 2.5 uses 1M theta
        }
    }
}

/// Turbo inference engine
pub struct TurboEngine {
    config: TurboConfig,
    kv_cache: ModelKVCache,
    layer_pin: LayerPin,
    // prefetcher: Option<AsyncPrefetcher>,
    device: Device,
    /// Secondary device for dual GPU mode
    secondary_device: Option<Device>,
    /// RoPE sin frequencies [max_seq_len, head_dim/2] - primary GPU
    rope_sin: Tensor,
    /// RoPE cos frequencies [max_seq_len, head_dim/2] - primary GPU
    rope_cos: Tensor,
    /// RoPE sin for secondary GPU (dual GPU mode)
    rope_sin_secondary: Option<Tensor>,
    /// RoPE cos for secondary GPU (dual GPU mode)
    rope_cos_secondary: Option<Tensor>,
}

impl TurboEngine {
    pub fn new(config: TurboConfig, device: Device) -> Self {
        Self::new_dual(config, device, None)
    }

    /// Create TurboEngine with optional secondary GPU for dual GPU mode
    pub fn new_dual(config: TurboConfig, device: Device, secondary_device: Option<Device>) -> Self {
        let kv_cache = ModelKVCache::new(config.n_layers, config.max_seq_len);
        let layer_pin = LayerPin::new(config.vram_budget_gb, config.pin_strategy.clone());

        // Compute RoPE frequencies for primary GPU
        let (rope_sin, rope_cos) = compute_rope_freqs(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            &device,
        ).expect("Failed to compute RoPE frequencies");

        // Compute RoPE for secondary GPU if dual GPU mode
        let (rope_sin_secondary, rope_cos_secondary) = if let Some(ref sec_dev) = secondary_device {
            let (sin2, cos2) = compute_rope_freqs(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta,
                sec_dev,
            ).expect("Failed to compute RoPE frequencies for secondary GPU");
            println!("🌀 RoPE initialized on BOTH GPUs: theta={}, head_dim={}, max_seq={}",
                     config.rope_theta, config.head_dim, config.max_seq_len);
            (Some(sin2), Some(cos2))
        } else {
            println!("🌀 RoPE initialized: theta={}, head_dim={}, max_seq={}",
                     config.rope_theta, config.head_dim, config.max_seq_len);
            (None, None)
        };

        Self {
            config,
            kv_cache,
            layer_pin,
            device,
            secondary_device,
            rope_sin,
            rope_cos,
            rope_sin_secondary,
            rope_cos_secondary,
        }
    }

    /// Check if a layer should run on secondary GPU (odd layers in dual GPU mode)
    #[inline]
    pub fn is_secondary_layer(&self, layer_idx: usize) -> bool {
        self.secondary_device.is_some() && layer_idx % 2 == 1
    }

    /// Get the appropriate device for a layer
    pub fn get_layer_device(&self, layer_idx: usize) -> &Device {
        if self.is_secondary_layer(layer_idx) {
            self.secondary_device.as_ref().unwrap()
        } else {
            &self.device
        }
    }

    /// Get RoPE tensors for a layer (returns tensors on correct device)
    fn get_rope_for_layer(&self, layer_idx: usize) -> (&Tensor, &Tensor) {
        if self.is_secondary_layer(layer_idx) {
            (self.rope_sin_secondary.as_ref().unwrap(),
             self.rope_cos_secondary.as_ref().unwrap())
        } else {
            (&self.rope_sin, &self.rope_cos)
        }
    }

    /// Process attention with KV-cache, RoPE, and Flash Attention
    pub fn attention_forward(
        &mut self,
        layer_idx: usize,
        x: &Tensor,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        wo: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        // Get current position offset from KV cache (for RoPE)
        let position_offset = self.kv_cache.get_layer(layer_idx).seq_len;

        // Get output dimensions from weight shapes: wq is [hidden, q_out], etc.
        let q_out = wq.dim(1)?;
        let kv_out = wk.dim(1)?;

        // Project Q, K, V - reshape for batched matmul: [B, S, H] -> [B*S, H]
        let x_flat = x.reshape((batch * seq_len, hidden))?;
        let q = x_flat.matmul(wq)?.reshape((batch, seq_len, q_out))?;
        let k = x_flat.matmul(wk)?.reshape((batch, seq_len, kv_out))?;
        let v = x_flat.matmul(wv)?.reshape((batch, seq_len, kv_out))?;

        // Reshape for multi-head attention (contiguous for matmul)
        let q = q.reshape((batch, seq_len, self.config.n_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?; // [batch, n_heads, seq_len, head_dim]
        let k = k.reshape((batch, seq_len, self.config.n_kv_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq_len, self.config.n_kv_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // 🌀 Apply RoPE to Q and K (using correct GPU's RoPE tensors!)
        let (rope_sin, rope_cos) = self.get_rope_for_layer(layer_idx);
        let q = apply_rope(&q, rope_sin, rope_cos, position_offset)?;
        let k = apply_rope(&k, rope_sin, rope_cos, position_offset)?;

        // Update KV cache with position-encoded K
        let cache = self.kv_cache.get_layer_mut(layer_idx);
        cache.append(&k, &v)?;

        // Get full K, V from cache (includes all previously cached tokens)
        let (full_k, full_v) = cache.get().unwrap();

        // Apply attention (with Flash Attention if enabled)
        let att_out = if self.config.use_flash_attention {
            flash_attention_forward(
                &q,
                full_k,
                full_v,
                &FlashAttentionConfig::default(),
            )?
        } else {
            // Standard attention (make tensors contiguous)
            let scale = 1.0 / (self.config.head_dim as f32).sqrt();
            let k_t = full_k.transpose(2, 3)?.contiguous()?;
            let att = (q.contiguous()?.matmul(&k_t)? * scale as f64)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.contiguous()?.matmul(&full_v.contiguous()?)?
        };

        // Reshape and project output (contiguous after transpose)
        let hidden_out = self.config.n_heads * self.config.head_dim;
        let att_out = att_out.transpose(1, 2)?.contiguous()?
            .reshape((batch, seq_len, hidden_out))?;

        // Output projection: [B, S, H] -> [B*S, H] -> matmul -> [B, S, H]
        let att_flat = att_out.reshape((batch * seq_len, hidden_out))?;
        let o_out = wo.dim(1)?;
        att_flat.matmul(wo)?.reshape((batch, seq_len, o_out))
    }

    /// Clear KV cache for new sequence
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.kv_cache.seq_len()
    }

    // ============ LAYER PINNING METHODS ============

    /// Check if a layer should be pinned based on strategy
    pub fn should_pin_layer(&self, layer_idx: usize) -> bool {
        self.layer_pin.should_pin(layer_idx, self.config.n_layers)
    }

    /// Get pinned tensor if available
    pub fn get_pinned(&self, layer_idx: usize, name: &str) -> Option<&Tensor> {
        self.layer_pin.get(layer_idx, name)
    }

    /// Pin a tensor (returns true if successful, false if VRAM budget exceeded)
    pub fn pin_tensor(&mut self, layer_idx: usize, name: &str, tensor: Tensor) -> bool {
        if self.layer_pin.should_pin(layer_idx, self.config.n_layers) {
            self.layer_pin.pin(layer_idx, name, tensor)
        } else {
            false
        }
    }

    /// Pin a tensor with automatic eviction if budget exceeded
    pub fn pin_tensor_smart(&mut self, layer_idx: usize, name: &str, tensor: Tensor) -> bool {
        if !self.layer_pin.should_pin(layer_idx, self.config.n_layers) {
            return false;
        }

        let size = tensor.elem_count() * tensor.dtype().size_in_bytes();
        let available = self.layer_pin.vram_budget.saturating_sub(self.layer_pin.vram_used);

        if size > available {
            // Try to evict to make room
            let needed = size - available;
            let freed = self.layer_pin.evict_until_space(needed);
            if freed < needed {
                // Still not enough space, skip pinning
                return false;
            }
        }

        self.layer_pin.pin(layer_idx, name, tensor)
    }

    /// Check if tensor is already pinned
    pub fn is_pinned(&self, layer_idx: usize, name: &str) -> bool {
        self.layer_pin.is_pinned(layer_idx, name)
    }

    /// Get VRAM usage for pinned layers
    pub fn pinned_vram_gb(&self) -> f32 {
        self.layer_pin.vram_usage_gb()
    }

    /// Get number of pinned layers
    pub fn pinned_layer_count(&self) -> usize {
        self.layer_pin.pinned_count()
    }

    /// Evict a specific layer
    pub fn evict_layer(&mut self, layer_idx: usize) -> usize {
        self.layer_pin.evict(layer_idx)
    }

    /// Clear all pinned layers
    pub fn clear_pinned(&mut self) {
        self.layer_pin.clear();
    }
}

// ============================================================================
// ENVIRONMENT SETUP
// ============================================================================

/// Set up CUDA environment for optimal performance
pub fn setup_cuda_env() {
    // Increase CUDA memory cache
    std::env::set_var("CUDA_CACHE_MAXSIZE", "2147483648"); // 2GB

    // Enable TF32 for faster matrix ops on Ampere+
    std::env::set_var("NVIDIA_TF32_OVERRIDE", "1");

    // Disable memory pool to reduce fragmentation
    // std::env::set_var("PYTORCH_NO_CUDA_MEMORY_CACHING", "1");

    println!("🚀 CUDA environment configured for turbo inference");
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache() {
        let mut cache = ModelKVCache::new(4, 1024);
        assert_eq!(cache.seq_len(), 0);

        // Would need device for actual tensor tests
    }

    #[test]
    fn test_pin_strategy() {
        let pin = LayerPin::new(4.0, PinStrategy::FirstLast { first: 2, last: 2 });
        assert!(pin.should_pin(0, 10)); // First
        assert!(pin.should_pin(1, 10)); // First
        assert!(!pin.should_pin(5, 10)); // Middle
        assert!(pin.should_pin(8, 10)); // Last
        assert!(pin.should_pin(9, 10)); // Last
    }
}
