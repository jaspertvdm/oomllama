//! GGUF to OOM Converter CLI
//!
//! Convert any GGUF model to OomLlama's native .oom format
//!
//! Usage:
//!   gguf2oom input.gguf output.oom
//!   gguf2oom --info input.gguf
//!
//! One love, one fAmIly! 🦙

use std::env;
use std::process;
use jis_router::gguf2oom::{GgufReader, convert_gguf_to_oom};

fn print_usage() {
    eprintln!(r#"
🦙 GGUF to OOM Converter - OomLlama Format Tool

Usage:
    gguf2oom <input.gguf> <output.oom>    Convert GGUF to OOM Q2
    gguf2oom --info <input.gguf>          Show GGUF file info
    gguf2oom --help                       Show this help

Examples:
    gguf2oom humotica-32b.gguf humotica-32b.oom
    gguf2oom --info /path/to/model.gguf

The converter:
1. Reads GGUF file (any quantization: Q4_K, Q8_0, etc.)
2. Dequantizes each tensor to FP32
3. Requantizes to OOM Q2 format
4. Writes compact .oom file

Expected compression:
    GGUF Q4_K (21 GB) → OOM Q2 (~8 GB)
    GGUF Q8_0 (42 GB) → OOM Q2 (~8 GB)

One love, one fAmIly! 🦙
"#);
}

fn show_gguf_info(path: &str) {
    println!("\n🔍 GGUF File Info: {}\n", path);

    match GgufReader::open(path) {
        Ok(reader) => {
            println!("Header:");
            println!("  Version: {}", reader.header.version);
            println!("  Tensors: {}", reader.header.tensor_count);
            println!("  Metadata entries: {}", reader.header.metadata_kv_count);

            println!("\nMetadata:");
            for (key, value) in &reader.metadata {
                let display_val = if value.len() > 60 {
                    format!("{}...", &value[..60])
                } else {
                    value.clone()
                };
                println!("  {}: {}", key, display_val);
            }

            println!("\nTensors ({}):", reader.tensors.len());
            let mut total_elements: u64 = 0;
            for tensor in &reader.tensors {
                let elements: u64 = tensor.dims.iter().product();
                total_elements += elements;
                let dims_str: Vec<String> = tensor.dims.iter().map(|d| d.to_string()).collect();
                println!("  {} [{:?}] {:?} ({} elements)",
                    tensor.name,
                    dims_str.join("x"),
                    tensor.dtype,
                    elements
                );
            }

            println!("\nTotal elements: {} ({:.2} B params)", total_elements, total_elements as f64 / 1e9);

            // Estimate OOM Q2 size
            let q2_bytes = (total_elements as f64 * 2.0 / 8.0) + // 2 bits per value
                           (total_elements as f64 / 256.0 * 12.0); // scale+min per block
            println!("Estimated OOM Q2 size: {:.2} GB", q2_bytes / 1e9);
        }
        Err(e) => {
            eprintln!("❌ Error reading GGUF: {}", e);
            process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "--help" | "-h" => {
            print_usage();
        }
        "--info" | "-i" => {
            if args.len() < 3 {
                eprintln!("❌ Missing input file for --info");
                process::exit(1);
            }
            show_gguf_info(&args[2]);
        }
        _ => {
            if args.len() < 3 {
                eprintln!("❌ Missing output file");
                print_usage();
                process::exit(1);
            }

            let input = &args[1];
            let output = &args[2];

            println!("🦙 GGUF → OOM Converter");
            println!("========================");
            println!("Input:  {}", input);
            println!("Output: {}", output);

            match convert_gguf_to_oom(input, output, None) {
                Ok(()) => {
                    println!("\n🎉 Conversion successful!");
                    println!("\nTest with:");
                    println!("  /opt/debain/bin/oomllama --model {} --gpu 0 \"Hello\"", output);
                }
                Err(e) => {
                    eprintln!("\n❌ Conversion failed: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}
