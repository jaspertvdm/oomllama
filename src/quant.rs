//! Q2 & Q4 Quantization Kernels
//! The "Oom" in OomLlama

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::Mmap;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct BlockQ2View<'a> {
    pub scale: f32,
    pub min: f32,
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockQ2View<'a> {
    pub fn dequantize(&self, dest: &mut [f32]) {
        for (byte_idx, &byte) in self.data.iter().enumerate() {
            for i in 0..4 {
                let value_idx = byte_idx * 4 + i;
                if value_idx >= self.num_values { break; }
                let shift = i * 2;
                let q = (byte >> shift) & 0b11;
                dest[value_idx] = self.min + (q as f32) * self.scale;
            }
        }
    }
}

pub struct BlockQ4View<'a> {
    pub scale: f32,
    pub min: f32,
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockQ4View<'a> {
    pub fn dequantize(&self, dest: &mut [f32]) {
        for (byte_idx, &byte) in self.data.iter().enumerate() {
            for i in 0..2 {
                let value_idx = byte_idx * 2 + i;
                if value_idx >= self.num_values { break; }
                let shift = i * 4;
                let q = (byte >> shift) & 0x0F;
                dest[value_idx] = self.min + (q as f32) * self.scale;
            }
        }
    }
}

pub struct BlockQ8View<'a> {
    pub scale: f32,
    pub min: f32,
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockQ8View<'a> {
    pub fn dequantize(&self, dest: &mut [f32]) {
        for (i, &q) in self.data.iter().enumerate() {
            if i >= self.num_values { break; }
            dest[i] = self.min + (q as f32) * self.scale;
        }
    }
}

pub struct BlockQ2 {
    pub scale: f32,
    pub min: f32,
    pub data: Vec<u8>,
    pub num_values: usize,
}

impl BlockQ2 {
    pub fn new(values: &[f32]) -> Self {
        let num_values = values.len();
        if num_values == 0 { return Self { scale: 1.0, min: 0.0, data: vec![], num_values: 0 }; }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in values {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let range = max - min;
        let scale = if range.abs() < 1e-9 { 0.0 } else { range / 3.0 };
        let mut data = Vec::with_capacity((num_values + 3) / 4);
        let mut current_byte: u8 = 0;
        let mut shift = 0;
        for &v in values {
            let q = if scale == 0.0 { 0 } else {
                let norm = (v - min) / scale;
                (norm.round() as u8).min(3)
            };
            current_byte |= q << shift;
            shift += 2;
            if shift == 8 {
                data.push(current_byte);
                current_byte = 0;
                shift = 0;
            }
        }
        if shift > 0 { data.push(current_byte); }
        Self { scale, min, data, num_values }
    }
}

pub struct BlockQ4 {
    pub scale: f32,
    pub min: f32,
    pub data: Vec<u8>,
    pub num_values: usize,
}

impl BlockQ4 {
    pub fn new(values: &[f32]) -> Self {
        let num_values = values.len();
        if num_values == 0 { return Self { scale: 1.0, min: 0.0, data: vec![], num_values: 0 }; }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in values {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let range = max - min;
        let scale = if range.abs() < 1e-9 { 0.0 } else { range / 15.0 };
        let mut data = Vec::with_capacity((num_values + 1) / 2);
        let mut current_byte: u8 = 0;
        let mut shift = 0;
        for &v in values {
            let q = if scale == 0.0 { 0 } else {
                let norm = (v - min) / scale;
                (norm.round() as u8).min(15)
            };
            current_byte |= q << shift;
            shift += 4;
            if shift == 8 {
                data.push(current_byte);
                current_byte = 0;
                shift = 0;
            }
        }
        if shift > 0 { data.push(current_byte); }
        Self { scale, min, data, num_values }
    }
}

pub struct OomTensorMeta {
    pub offset: usize,
    pub num_blocks: u32,
    pub total_values: u32,
    pub quant_type: u8,
}

pub struct OomLoader {
    mmap: Mmap,
    pub tensors: HashMap<String, OomTensorMeta>,
}

impl OomLoader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut offset = 0;
        
        if mmap.len() < 12 { return Err("File too small".into()); }
        if &mmap[offset..offset+4] != b"OOML" { return Err("Invalid OomLlama file format".into()); }
        offset += 4;
        
        let _version = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        offset += 4;
        
        let num_tensors = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        offset += 4;
        
        let mut tensors = HashMap::new();
        println!("🔍 Scanning .oom metadata ({} tensors)...", num_tensors);

        for _ in 0..num_tensors {
            if offset + 4 > mmap.len() { break; }
            let name_len = u32::from_le_bytes(mmap[offset..offset+4].try_into()?) as usize;
            offset += 4;
            
            let name = String::from_utf8(mmap[offset..offset+name_len].to_vec())?;
            offset += name_len;
            
            let quant_type = mmap[offset];
            offset += 1;
            
            let num_blocks = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;
            
            let total_values = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;
            
            tensors.insert(name, OomTensorMeta { offset, num_blocks, total_values, quant_type });
            
            // Skip block data to get to next tensor metadata quickly
            for _ in 0..num_blocks {
                if offset + 12 > mmap.len() { break; }
                let data_len = u32::from_le_bytes(mmap[offset+8..offset+12].try_into()?) as usize;
                offset += 12 + data_len;
            }
        }
        
        println!("✅ Metadata loaded. Ready for lazy extraction.");
        Ok(Self { mmap, tensors })
    }

    /// Dequantize a specific block of a tensor. Used for fine-grained lazy loading.
    pub fn dequantize_block(&self, tensor_name: &str, block_idx: u32, dest: &mut [f32]) -> Result<()> {
        let meta = self.tensors.get(tensor_name).ok_or_else(|| {
            format!("Tensor not found: '{}' (available: {} tensors)", tensor_name, self.tensors.len())
        })?;
        if block_idx >= meta.num_blocks { return Err("Block index out of bounds".into()); }

        let mut offset = meta.offset;
        // Find block offset (this is O(N) blocks, but still faster than full dequant)
        for _ in 0..block_idx {
            let data_len = u32::from_le_bytes(self.mmap[offset+8..offset+12].try_into()?) as usize;
            offset += 12 + data_len;
        }

        let scale = f32::from_le_bytes(self.mmap[offset..offset+4].try_into()?);
        let min = f32::from_le_bytes(self.mmap[offset+4..offset+8].try_into()?);
        let data_len = u32::from_le_bytes(self.mmap[offset+8..offset+12].try_into()?) as usize;
        offset += 12;

        let num_vals = dest.len().min(256);
        if meta.quant_type == 32 {
            // F32: direct copy
            for i in 0..num_vals {
                dest[i] = f32::from_le_bytes(self.mmap[offset + i * 4..offset + (i + 1) * 4].try_into()?);
            }
        } else if meta.quant_type == 8 {
            let view = BlockQ8View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
            view.dequantize(&mut dest[..num_vals]);
        } else if meta.quant_type == 4 {
            let view = BlockQ4View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
            view.dequantize(&mut dest[..num_vals]);
        } else {
            let view = BlockQ2View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
            view.dequantize(&mut dest[..num_vals]);
        }

        Ok(())
    }

    /// Dequantize entire tensor into a provided buffer.
    pub fn dequantize_tensor_into(&self, name: &str, dest: &mut [f32]) -> Result<()> {
        let meta = self.tensors.get(name).ok_or_else(|| {
            format!("Tensor not found: '{}' (available: {} tensors)", name, self.tensors.len())
        })?;
        if dest.len() < meta.total_values as usize { return Err("Destination buffer too small".into()); }

        let mut offset = meta.offset;
        let mut current_pos = 0;
        let mut nan_blocks = 0;
        for block_idx in 0..meta.num_blocks {
            let scale = f32::from_le_bytes(self.mmap[offset..offset+4].try_into()?);
            let min = f32::from_le_bytes(self.mmap[offset+4..offset+8].try_into()?);
            let data_len = u32::from_le_bytes(self.mmap[offset+8..offset+12].try_into()?) as usize;
            offset += 12;

            // Sanitize NaN/inf in scale/min - set to 0 if corrupted
            let (scale, min) = if scale.is_nan() || min.is_nan() || scale.is_infinite() || min.is_infinite() {
                nan_blocks += 1;
                (0.0_f32, 0.0_f32)  // Corrupt block → all zeros
            } else {
                (scale, min)
            };

            let num_vals = (meta.total_values as usize - current_pos).min(256);
            if meta.quant_type == 32 {
                // F32: direct copy
                for i in 0..num_vals {
                    dest[current_pos + i] = f32::from_le_bytes(self.mmap[offset + i * 4..offset + (i + 1) * 4].try_into()?);
                }
            } else if meta.quant_type == 8 {
                let view = BlockQ8View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize(&mut dest[current_pos..current_pos + num_vals]);
            } else if meta.quant_type == 4 {
                let view = BlockQ4View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize(&mut dest[current_pos..current_pos + num_vals]);
            } else {
                let view = BlockQ2View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize(&mut dest[current_pos..current_pos + num_vals]);
            }
            offset += data_len;
            current_pos += num_vals;
        }
        // Only warn once per session about corrupt blocks (suppress repeat warnings)
        // We use a static HashSet to track which tensors we've warned about
        use std::sync::OnceLock;
        use std::collections::HashSet;
        use std::sync::Mutex;
        static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

        if nan_blocks > 0 {
            let warned = WARNED.get_or_init(|| Mutex::new(HashSet::new()));
            let mut warned_set = warned.lock().unwrap();
            if !warned_set.contains(name) {
                println!("⚠️ {} had {}/{} corrupt blocks (sanitized)", name, nan_blocks, meta.num_blocks);
                warned_set.insert(name.to_string());
            }
        }
        Ok(())
    }

    pub fn dequantize_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let meta = self.tensors.get(name).ok_or_else(|| {
            format!("Tensor not found: '{}' (available: {} tensors)", name, self.tensors.len())
        })?;
        let mut result = vec![0.0; meta.total_values as usize];
        self.dequantize_tensor_into(name, &mut result)?;
        Ok(result)
    }
}