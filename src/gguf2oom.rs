//! GGUF to OOM Converter
//! Convert any GGUF model to OomLlama's .oom format
//!
//! GGUF Format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//! OOM Format: /srv/jtel-stack/sandbox/ai/codex/oomllama_format_spec.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// GGUF Magic and constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const OOM_MAGIC: &[u8; 4] = b"OOML";
const OOM_VERSION: u32 = 1;
const BLOCK_SIZE: usize = 256;

// GGUF Tensor Types (from ggml.h)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
    Unknown = 255,
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            29 => GgmlType::BF16,
            _ => GgmlType::Unknown,
        }
    }
}

// GGUF Value Types
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct GgufReader {
    mmap: Mmap,
    pub header: GgufHeader,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_data_offset: usize,
    pub metadata: HashMap<String, String>,
}

impl GgufReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        println!("📖 Reading GGUF file ({:.2} GB)...", mmap.len() as f64 / 1e9);

        let mut offset = 0;

        // Read magic
        let magic = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        if magic != GGUF_MAGIC {
            return Err(format!("Invalid GGUF magic: 0x{:08X} (expected 0x{:08X})", magic, GGUF_MAGIC).into());
        }
        offset += 4;

        // Read version
        let version = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        offset += 4;
        println!("   Version: {}", version);

        // Read counts
        let tensor_count = u64::from_le_bytes(mmap[offset..offset+8].try_into()?);
        offset += 8;
        let metadata_kv_count = u64::from_le_bytes(mmap[offset..offset+8].try_into()?);
        offset += 8;

        println!("   Tensors: {}", tensor_count);
        println!("   Metadata entries: {}", metadata_kv_count);

        let header = GgufHeader { version, tensor_count, metadata_kv_count };

        // Skip metadata (we don't need it for conversion, but parse to find tensor info)
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            // Read key
            let key_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
            offset += 8;
            let key = String::from_utf8_lossy(&mmap[offset..offset+key_len]).to_string();
            offset += key_len;

            // Read value type
            let value_type = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;

            // Skip value based on type
            match value_type {
                0 => offset += 1,  // uint8
                1 => offset += 1,  // int8
                2 => offset += 2,  // uint16
                3 => offset += 2,  // int16
                4 => offset += 4,  // uint32
                5 => offset += 4,  // int32
                6 => offset += 4,  // float32
                7 => offset += 1,  // bool
                8 => {  // string
                    let str_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
                    offset += 8;
                    let value = String::from_utf8_lossy(&mmap[offset..offset+str_len]).to_string();
                    offset += str_len;
                    metadata.insert(key.clone(), value);
                }
                9 => {  // array
                    let arr_type = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
                    offset += 4;
                    let arr_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
                    offset += 8;
                    // Skip array elements
                    let elem_size = match arr_type {
                        0 | 1 | 7 => 1,
                        2 | 3 => 2,
                        4 | 5 | 6 => 4,
                        10 | 11 | 12 => 8,
                        8 => {
                            // Array of strings - need to skip each
                            for _ in 0..arr_len {
                                let slen = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
                                offset += 8 + slen;
                            }
                            0
                        }
                        _ => 0,
                    };
                    if elem_size > 0 {
                        offset += arr_len * elem_size;
                    }
                }
                10 | 11 | 12 => offset += 8,  // uint64, int64, float64
                _ => {}
            }
        }

        // Read tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            // Read name
            let name_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
            offset += 8;
            let name = String::from_utf8_lossy(&mmap[offset..offset+name_len]).to_string();
            offset += name_len;

            // Read dimensions
            let n_dims = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(u64::from_le_bytes(mmap[offset..offset+8].try_into()?));
                offset += 8;
            }

            // Read type
            let dtype = GgmlType::from(u32::from_le_bytes(mmap[offset..offset+4].try_into()?));
            offset += 4;

            // Read offset
            let tensor_offset = u64::from_le_bytes(mmap[offset..offset+8].try_into()?);
            offset += 8;

            tensors.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        // Align to 32 bytes for tensor data
        let tensor_data_offset = (offset + 31) & !31;

        println!("   Tensor data starts at: 0x{:X}", tensor_data_offset);

        Ok(Self { mmap, header, tensors, tensor_data_offset, metadata })
    }

    /// Get total number of elements in a tensor
    fn tensor_elements(&self, info: &GgufTensorInfo) -> u64 {
        info.dims.iter().product()
    }

    /// Dequantize a tensor to FP32
    pub fn dequantize_tensor(&self, info: &GgufTensorInfo) -> Result<Vec<f32>> {
        let n_elements = self.tensor_elements(info) as usize;
        let data_offset = self.tensor_data_offset + info.offset as usize;

        let mut result = vec![0.0f32; n_elements];

        match info.dtype {
            GgmlType::F32 => {
                // Direct copy
                for i in 0..n_elements {
                    let off = data_offset + i * 4;
                    result[i] = f32::from_le_bytes(self.mmap[off..off+4].try_into()?);
                }
            }
            GgmlType::F16 => {
                // FP16 to FP32
                for i in 0..n_elements {
                    let off = data_offset + i * 2;
                    let bits = u16::from_le_bytes(self.mmap[off..off+2].try_into()?);
                    result[i] = f16_to_f32(bits);
                }
            }
            GgmlType::BF16 => {
                // BF16 to FP32
                for i in 0..n_elements {
                    let off = data_offset + i * 2;
                    let bits = u16::from_le_bytes(self.mmap[off..off+2].try_into()?);
                    result[i] = bf16_to_f32(bits);
                }
            }
            GgmlType::Q4_0 => {
                dequant_q4_0(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q4_K | GgmlType::Q4_1 => {
                dequant_q4_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q8_0 => {
                dequant_q8_0(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q6_K => {
                dequant_q6_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q5_K => {
                dequant_q5_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q2_K => {
                dequant_q2_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q3_K => {
                dequant_q3_k(&self.mmap[data_offset..], &mut result)?;
            }
            _ => {
                return Err(format!("Unsupported tensor type: {:?}", info.dtype).into());
            }
        }

        Ok(result)
    }
}

// FP16 to FP32 conversion
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormal
        let mut m = mant;
        let mut e = -14i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_mant = m << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_mant)
    } else if exp == 31 {
        // Inf/NaN
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        // Normal
        let f32_exp = ((exp - 15 + 127) as u32) << 23;
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_mant)
    }
}

// BF16 to FP32 conversion (simple - just shift)
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// Q4_0 dequantization (block size 32)
fn dequant_q4_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let n_blocks = output.len() / block_size;
    let bytes_per_block = 2 + block_size / 2; // scale (f16) + 16 bytes of q4

    for i in 0..n_blocks {
        let block_offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[block_offset..block_offset+2].try_into()?));

        for j in 0..16 {
            let byte = data[block_offset + 2 + j];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            output[i * block_size + j * 2] = scale * q0 as f32;
            output[i * block_size + j * 2 + 1] = scale * q1 as f32;
        }
    }
    Ok(())
}

// Q4_K dequantization (block size 256, super-blocks)
// Based on llama.cpp ggml-quants.c implementation
fn dequant_q4_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;  // QK_K
    let n_super_blocks = output.len() / super_block_size;

    // Q4_K structure: d (f16), dmin (f16), scales (12 bytes), qs (128 bytes) = 144 bytes
    let bytes_per_super_block = 144;

    // Helper to extract scale and min from packed format (llama.cpp get_scale_min_k4)
    fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
        if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (sc, m)
        }
    }

    for sb in 0..n_super_blocks {
        let sb_offset = sb * bytes_per_super_block;

        if sb_offset + bytes_per_super_block > data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes(data[sb_offset..sb_offset+2].try_into()?));
        let dmin = f16_to_f32(u16::from_le_bytes(data[sb_offset+2..sb_offset+4].try_into()?));

        let scales = &data[sb_offset + 4..sb_offset + 16];
        let qs = &data[sb_offset + 16..sb_offset + 144];

        // Process 4 pairs of sub-blocks (8 total, 32 values each = 256)
        let mut out_idx = sb * super_block_size;
        let mut q_idx = 0;

        for j in 0..4 {  // QK_K/64 = 4 iterations
            // Get scales and mins for 2 sub-blocks
            let (sc1, m1) = get_scale_min_k4(j * 2, scales);
            let (sc2, m2) = get_scale_min_k4(j * 2 + 1, scales);

            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            // Each iteration processes 64 values (2 sub-blocks of 32)
            for l in 0..32 {
                let q_byte = qs[q_idx + l];
                // Lower nibble -> first sub-block
                output[out_idx + l] = d1 * (q_byte & 0xF) as f32 - min1;
                // Upper nibble -> second sub-block
                output[out_idx + 32 + l] = d2 * (q_byte >> 4) as f32 - min2;
            }

            out_idx += 64;
            q_idx += 32;
        }
    }
    Ok(())
}

// Q8_0 dequantization (block size 32)
fn dequant_q8_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let n_blocks = output.len() / block_size;
    let bytes_per_block = 2 + block_size; // scale (f16) + 32 bytes of q8

    for i in 0..n_blocks {
        let block_offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[block_offset..block_offset+2].try_into()?));

        for j in 0..block_size {
            let q = data[block_offset + 2 + j] as i8;
            output[i * block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

// Q6_K dequantization - proper implementation based on llama.cpp
fn dequant_q6_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    // Q6_K block layout (210 bytes for 256 values):
    // - ql[128]: lower 4 bits of each quant (256 values * 4 bits / 8 = 128 bytes)
    // - qh[64]:  upper 2 bits of each quant (256 values * 2 bits / 8 = 64 bytes)
    // - scales[16]: per-sub-block scales (16 sub-blocks, signed i8)
    // - d[2]: super-block scale (f16)
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 210;

    let n_blocks = output.len() / QK_K;

    for i in 0..n_blocks {
        let block_start = i * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            // Fill remainder with zeros
            for j in i * QK_K..output.len() {
                output[j] = 0.0;
            }
            break;
        }

        let block = &data[block_start..block_start + BLOCK_SIZE];

        // Parse block layout
        let ql = &block[0..128];       // low 4 bits
        let qh = &block[128..192];     // high 2 bits
        let scales = &block[192..208]; // sub-block scales (signed i8)
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        let out_base = i * QK_K;

        // Process in two halves of 128 values each
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;

            for l in 0..32 {
                // Which sub-block within this half (0 or 1)
                let is = l / 16;

                // Read quantized bytes
                let ql0 = ql[ql_off + l];
                let ql32 = ql[ql_off + l + 32];
                let qh_byte = qh[qh_off + l];

                // Extract 6-bit values: combine low 4 bits + high 2 bits, subtract 32 to center
                let q1 = (((ql0 & 0x0F) | ((qh_byte & 0x03) << 4)) as i8).wrapping_sub(32);
                let q2 = (((ql32 & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i8).wrapping_sub(32);
                let q3 = (((ql0 >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i8).wrapping_sub(32);
                let q4 = (((ql32 >> 4) | ((qh_byte >> 6) << 4)) as i8).wrapping_sub(32);

                // Get scales for each output position (signed i8)
                let sc0 = scales[sc_off + is] as i8 as f32;
                let sc1 = scales[sc_off + is + 2] as i8 as f32;
                let sc2 = scales[sc_off + is + 4] as i8 as f32;
                let sc3 = scales[sc_off + is + 6] as i8 as f32;

                // Write outputs
                let base = out_base + half * 128;
                output[base + l] = d * sc0 * q1 as f32;
                output[base + l + 32] = d * sc1 * q2 as f32;
                output[base + l + 64] = d * sc2 * q3 as f32;
                output[base + l + 96] = d * sc3 * q4 as f32;
            }
        }
    }
    Ok(())
}

// Q5_K dequantization (simplified)
fn dequant_q5_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 176;

    for i in 0..n_blocks {
        let offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[offset..offset+2].try_into().unwrap_or([0,0])));

        for j in 0..super_block_size {
            let q = ((data.get(offset + 4 + j / 2).unwrap_or(&0) >> ((j % 2) * 4)) & 0x1F) as i8 - 16;
            output[i * super_block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

// Q2_K dequantization (simplified)
fn dequant_q2_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 84;

    for i in 0..n_blocks {
        let offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[offset..offset+2].try_into().unwrap_or([0,0])));

        for j in 0..super_block_size {
            let byte_idx = j / 4;
            let bit_shift = (j % 4) * 2;
            let q = ((data.get(offset + 4 + byte_idx).unwrap_or(&0) >> bit_shift) & 0x03) as i8 - 2;
            output[i * super_block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

// Q3_K dequantization (simplified)
fn dequant_q3_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 110;

    for i in 0..n_blocks {
        let offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[offset..offset+2].try_into().unwrap_or([0,0])));

        for j in 0..super_block_size {
            let q = ((data.get(offset + 4 + j / 3).unwrap_or(&0) >> ((j % 3) * 2)) & 0x07) as i8 - 4;
            output[i * super_block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

/// OOM Writer - Write tensors in OomLlama format
pub struct OomWriter {
    writer: BufWriter<File>,
    tensor_count: u32,
    tensor_data: Vec<u8>,
}

impl OomWriter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            tensor_count: 0,
            tensor_data: Vec::new(),
        })
    }

    /// Write header (called after all tensors added)
    fn write_header(&mut self) -> Result<()> {
        // Magic
        self.writer.write_all(OOM_MAGIC)?;
        // Version
        self.writer.write_all(&OOM_VERSION.to_le_bytes())?;
        // Tensor count
        self.writer.write_all(&self.tensor_count.to_le_bytes())?;
        Ok(())
    }

    /// Add a tensor (quantizes to Q2)
    pub fn add_tensor_q2(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        // name_len (u32) + name + quant_type (u8) + num_blocks (u32) + total_values (u32)
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(2); // Q2
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Find min/max (skip NaN/inf values)
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            let mut valid_count = 0;
            for &v in block_values {
                if v.is_finite() {
                    if v < min { min = v; }
                    if v > max { max = v; }
                    valid_count += 1;
                }
            }

            // Handle blocks with no valid values or empty blocks
            if valid_count == 0 || min > max {
                min = 0.0;
                max = 0.0;
            }

            let range = max - min;
            let scale = if range.abs() < 1e-9 { 0.0 } else { range / 3.0 };

            // Quantize
            let mut qdata = Vec::with_capacity((block_values.len() + 3) / 4);
            let mut current_byte: u8 = 0;
            let mut shift = 0;

            for &v in block_values {
                let q = if scale == 0.0 || !v.is_finite() { 0 } else {
                    let norm = (v - min) / scale;
                    (norm.round().clamp(0.0, 3.0) as u8)
                };
                current_byte |= q << shift;
                shift += 2;
                if shift == 8 {
                    qdata.push(current_byte);
                    current_byte = 0;
                    shift = 0;
                }
            }
            if shift > 0 { qdata.push(current_byte); }

            // Write block: scale (f32) + min (f32) + data_len (u32) + qdata
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(qdata.len() as u32).to_le_bytes());
            self.tensor_data.extend_from_slice(&qdata);
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Add a tensor (quantizes to Q4 - 4 bits per value, 16 levels)
    pub fn add_tensor_q4(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(4); // Q4
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Find min/max (skip NaN/inf values)
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            let mut valid_count = 0;
            for &v in block_values {
                if v.is_finite() {
                    if v < min { min = v; }
                    if v > max { max = v; }
                    valid_count += 1;
                }
            }

            if valid_count == 0 || min > max {
                min = 0.0;
                max = 0.0;
            }

            let range = max - min;
            let scale = if range.abs() < 1e-9 { 0.0 } else { range / 15.0 };

            // Quantize to 4 bits (2 values per byte)
            let mut qdata = Vec::with_capacity((block_values.len() + 1) / 2);
            let mut current_byte: u8 = 0;
            let mut shift = 0;

            for &v in block_values {
                let q = if scale == 0.0 || !v.is_finite() { 0 } else {
                    let norm = (v - min) / scale;
                    (norm.round().clamp(0.0, 15.0) as u8)
                };
                current_byte |= q << shift;
                shift += 4;
                if shift == 8 {
                    qdata.push(current_byte);
                    current_byte = 0;
                    shift = 0;
                }
            }
            if shift > 0 { qdata.push(current_byte); }

            // Write block: scale (f32) + min (f32) + data_len (u32) + qdata
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(qdata.len() as u32).to_le_bytes());
            self.tensor_data.extend_from_slice(&qdata);
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Add a tensor (quantizes to Q8 - 8 bits per value, 256 levels)
    /// Use for sensitive tensors like lm_head and embed_tokens
    pub fn add_tensor_q8(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(8); // Q8
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Find min/max (skip NaN/inf values)
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            let mut valid_count = 0;
            for &v in block_values {
                if v.is_finite() {
                    if v < min { min = v; }
                    if v > max { max = v; }
                    valid_count += 1;
                }
            }

            if valid_count == 0 || min > max {
                min = 0.0;
                max = 0.0;
            }

            let range = max - min;
            let scale = if range.abs() < 1e-9 { 0.0 } else { range / 255.0 };

            // Quantize to 8 bits (1 value per byte)
            let mut qdata = Vec::with_capacity(block_values.len());

            for &v in block_values {
                let q = if scale == 0.0 || !v.is_finite() { 0 } else {
                    let norm = (v - min) / scale;
                    (norm.round().clamp(0.0, 255.0) as u8)
                };
                qdata.push(q);
            }

            // Write block: scale (f32) + min (f32) + data_len (u32) + qdata
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(qdata.len() as u32).to_le_bytes());
            self.tensor_data.extend_from_slice(&qdata);
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Add a tensor (stores as F32 - no quantization)
    /// Use for debugging or when precision is critical
    pub fn add_tensor_f32(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(32); // F32 marker
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks (same format as Q4/Q8 but with raw F32 data)
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Write block header (scale and min are unused for F32, but kept for format compatibility)
            self.tensor_data.extend_from_slice(&(0.0f32).to_le_bytes()); // scale (unused)
            self.tensor_data.extend_from_slice(&(0.0f32).to_le_bytes()); // min (unused)

            // Data length is num_values * 4 bytes
            let data_len = block_values.len() * 4;
            self.tensor_data.extend_from_slice(&(data_len as u32).to_le_bytes());

            // Write raw F32 values
            for &v in block_values {
                self.tensor_data.extend_from_slice(&v.to_le_bytes());
            }
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Finalize and write the file
    pub fn finish(mut self) -> Result<()> {
        self.write_header()?;
        self.writer.write_all(&self.tensor_data)?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Map GGUF tensor names to HuggingFace/Candle format
fn map_tensor_name(gguf_name: &str) -> String {
    // Global tensors
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }

    // Layer tensors: blk.N.* -> model.layers.N.*
    if gguf_name.starts_with("blk.") {
        // Extract layer number
        let parts: Vec<&str> = gguf_name.split('.').collect();
        if parts.len() >= 3 {
            let layer_num = parts[1];
            let rest = &parts[2..].join(".");

            let mapped = match rest.as_str() {
                // Attention
                "attn_norm.weight" => format!("model.layers.{}.input_layernorm.weight", layer_num),
                "attn_q.weight" => format!("model.layers.{}.self_attn.q_proj.weight", layer_num),
                "attn_k.weight" => format!("model.layers.{}.self_attn.k_proj.weight", layer_num),
                "attn_v.weight" => format!("model.layers.{}.self_attn.v_proj.weight", layer_num),
                "attn_output.weight" => format!("model.layers.{}.self_attn.o_proj.weight", layer_num),
                "attn_q.bias" => format!("model.layers.{}.self_attn.q_proj.bias", layer_num),
                "attn_k.bias" => format!("model.layers.{}.self_attn.k_proj.bias", layer_num),
                "attn_v.bias" => format!("model.layers.{}.self_attn.v_proj.bias", layer_num),
                // FFN / MLP
                "ffn_norm.weight" => format!("model.layers.{}.post_attention_layernorm.weight", layer_num),
                "ffn_gate.weight" => format!("model.layers.{}.mlp.gate_proj.weight", layer_num),
                "ffn_up.weight" => format!("model.layers.{}.mlp.up_proj.weight", layer_num),
                "ffn_down.weight" => format!("model.layers.{}.mlp.down_proj.weight", layer_num),
                // Fallback
                _ => return gguf_name.to_string(),
            };
            return mapped;
        }
    }

    // Fallback: return as-is
    gguf_name.to_string()
}

/// Convert GGUF to OOM
pub fn convert_gguf_to_oom<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    progress_callback: Option<Box<dyn Fn(usize, usize, &str)>>,
) -> Result<()> {
    let reader = GgufReader::open(&input_path)?;
    let mut writer = OomWriter::create(&output_path)?;

    let total_tensors = reader.tensors.len();
    println!("\n🔄 Converting {} tensors to OOM Q8 format (8 bits = 256 levels)...", total_tensors);

    // First pass: check if embed_tokens needs to be replaced by lm_head (tied embeddings fallback)
    let mut embed_tokens_data: Option<Vec<f32>> = None;
    let mut lm_head_data: Option<Vec<f32>> = None;
    let mut use_lm_head_for_embeddings = false;

    // Debug: print first 3 tensor names to verify
    let first_names: Vec<&str> = reader.tensors.iter().take(3).map(|t| t.name.as_str()).collect();
    println!("\n   🔍 First 3 GGUF tensor names: {:?}\n", first_names);

    for tensor_info in reader.tensors.iter() {
        if tensor_info.name == "token_embd.weight" {
            println!("   🔍 Found token_embd.weight, checking values...");
            let data = reader.dequantize_tensor(tensor_info)?;
            let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("   📊 token_embd: max_abs={:.6}, min={:.6}, max={:.6}", max_abs, min_val, max_val);
            if max_abs < 0.001 {
                println!("   ⚠️ token_embd.weight has tiny values - will use output.weight");
                use_lm_head_for_embeddings = true;
            }
            embed_tokens_data = Some(data);
        } else if tensor_info.name == "output.weight" {
            println!("   🔍 Found output.weight, caching...");
            lm_head_data = Some(reader.dequantize_tensor(tensor_info)?);
        }
    }

    for (idx, tensor_info) in reader.tensors.iter().enumerate() {
        // Map GGUF tensor name to HuggingFace/Candle format
        let mapped_name = map_tensor_name(&tensor_info.name);

        if let Some(ref cb) = progress_callback {
            cb(idx + 1, total_tensors, &mapped_name);
        } else {
            print!("\r   [{}/{}] {}                    ", idx + 1, total_tensors, &mapped_name);
            std::io::stdout().flush()?;
        }

        // Skip non-weight tensors (like position embeddings that should stay FP32)
        // For now, convert everything

        // Dequantize to FP32 (or use cached data for embed/lm_head)
        let fp32_values = if tensor_info.name == "token_embd.weight" {
            if use_lm_head_for_embeddings {
                // Use lm_head weights for embeddings (tied embeddings fallback)
                println!("   🔄 Using output.weight for model.embed_tokens.weight (tied embeddings)");
                lm_head_data.clone().unwrap_or_else(|| reader.dequantize_tensor(tensor_info).unwrap())
            } else {
                embed_tokens_data.clone().unwrap_or_else(|| reader.dequantize_tensor(tensor_info).unwrap())
            }
        } else if tensor_info.name == "output.weight" {
            lm_head_data.clone().unwrap_or_else(|| reader.dequantize_tensor(tensor_info).unwrap())
        } else {
            reader.dequantize_tensor(tensor_info)?
        };

        // Debug: check for extreme values that suggest dequant issues
        if mapped_name.contains("v_proj") || mapped_name.contains("attn_v") {
            let min_val = fp32_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = fp32_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("   ⚠️ {} (type {:?}): min={:.4}, max={:.4}",
                     &tensor_info.name, tensor_info.dtype, min_val, max_val);
        }

        // Use Q8 quantization for best quality with reasonable compression
        // Q8 = 8 bits per value, ~4x smaller than F32
        writer.add_tensor_q8(&mapped_name, &fp32_values)?;
    }

    println!("\n✅ Writing OOM file...");
    writer.finish()?;

    // Print size comparison
    let input_size = std::fs::metadata(&input_path)?.len();
    let output_size = std::fs::metadata(&output_path)?.len();
    let ratio = input_size as f64 / output_size as f64;

    println!("\n📊 Conversion complete!");
    println!("   Input:  {:.2} GB (GGUF)", input_size as f64 / 1e9);
    println!("   Output: {:.2} GB (OOM Q8)", output_size as f64 / 1e9);
    println!("   Ratio:  {:.1}x smaller", ratio);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // Test some known values
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001); // 1.0
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.001); // 0.0
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001); // -1.0
    }
}
