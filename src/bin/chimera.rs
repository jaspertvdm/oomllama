//! CHIMERA-SCANNER CLI (batch mode).
//!
//! Usage:
//!   chimera scan <path>              # Scan without embeddings
//!   chimera scan <path> --embed      # Scan WITH GPU embeddings
//!   chimera scan <path> --embed --batch 500  # Stream mode: write every 500 files
//!   chimera report <path>
//!   chimera diff <path>

use std::env;
use std::path::{Path, PathBuf};
use std::io::{self, Write};

use jis_router::batch::{BatchConfig, BatchProcessor, write_vectors_jsonl};

// Force flush stdout for nohup compatibility
macro_rules! println_flush {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        let _ = io::stdout().flush();
    }};
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: chimera <scan|report|diff> <path> [--embed] [--batch N]");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --embed    Enable GPU embeddings (requires CUDA)");
        eprintln!("  --gpu N    Use GPU N for embeddings (default: 1)");
        eprintln!("  --batch N  Stream mode: write to disk every N files (saves memory!)");
        std::process::exit(1);
    }
    let command = &args[1];
    let path = PathBuf::from(&args[2]);

    // Parse flags
    let enable_embed = args.iter().any(|a| a == "--embed");
    let gpu_index = args.iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let batch_size = args.iter()
        .position(|a| a == "--batch")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0usize);  // 0 = no streaming (old behavior)

    match command.as_str() {
        "scan" => run_scan(&path, false, enable_embed, gpu_index, batch_size),
        "report" => run_scan(&path, true, enable_embed, gpu_index, batch_size),
        "diff" => run_diff(&path, enable_embed, gpu_index, batch_size),
        _ => {
            eprintln!("Unknown command: {}", command);
            std::process::exit(1);
        }
    }
}

fn run_scan(path: &Path, print_report: bool, enable_embed: bool, gpu_index: usize, batch_size: usize) {
    let cfg = BatchConfig {
        enable_embeddings: enable_embed,
        gpu_index,
        ..BatchConfig::default()
    };

    if enable_embed {
        println_flush!("🚀 GPU embeddings enabled on GPU {}", gpu_index);
    }
    if batch_size > 0 {
        println_flush!("📦 Streaming mode: writing every {} files", batch_size);
    }

    let batch = BatchProcessor::new(cfg);
    println_flush!("📂 Starting file collection from: {}", path.display());
    let files = collect_files(path);
    let total_files = files.len();
    println_flush!("📂 File collection complete: {} files", total_files);

    // Streaming mode: write batches incrementally to save memory
    if batch_size > 0 {
        let tibet = jis_router::tibet::TibetFactory::new("chimera-cli");
        let mut batch_num = 0;
        let mut total_chunks = 0;
        let mut total_embedded = 0;
        let mut batch_vectors = Vec::new();

        println_flush!("🔍 Collected {} files, starting processing...", total_files);

        for (i, file) in files.iter().enumerate() {
            println_flush!("📄 [{}/{}] Processing: {}", i + 1, total_files, file.display());

            match batch.process_path(file, "BatchScan") {
                Ok((_report, vectors)) => {
                    if !vectors.is_empty() {
                        batch_vectors.extend(vectors);
                    }
                }
                Err(e) => {
                    println_flush!("  ⚠️ Error: {}", e);
                }
            }

            // Write batch when we hit batch_size or at the end
            if (i + 1) % batch_size == 0 || i + 1 == total_files {
                if !batch_vectors.is_empty() {
                    batch_num += 1;
                    let embedded = batch_vectors.iter().filter(|v| !v.vector.is_empty()).count();
                    total_embedded += embedded;
                    total_chunks += batch_vectors.len();

                    let batch_id = format!("CHIM-STREAM-{:04}", batch_num);
                    let out_path = PathBuf::from("data/kmbit/batches").join(format!("{}.jsonl", batch_id));
                    let _ = write_vectors_jsonl(&out_path, &batch_vectors);

                    println_flush!("  💾 Batch {} written: {} chunks ({} embedded)",
                             batch_num, batch_vectors.len(), embedded);

                    // Clear memory!
                    batch_vectors.clear();
                    batch_vectors.shrink_to_fit();
                }
            }
        }

        println_flush!("\n✅ CHIMERA streaming complete!");
        println_flush!("   Total files: {}", total_files);
        println_flush!("   Total batches: {}", batch_num);
        println_flush!("   Total chunks: {}", total_chunks);
        if enable_embed {
            println_flush!("   Total embedded: {} (384 dimensions each)", total_embedded);
        }
        return;
    }

    // Original non-streaming mode
    let mut all_vectors = Vec::new();
    let mut all_items = Vec::new();

    for (i, file) in files.iter().enumerate() {
        if enable_embed {
            println!("📄 [{}/{}] Processing: {}", i + 1, total_files, file.display());
        }
        if let Ok((report, vectors)) = batch.process_path(file, "BatchScan") {
            all_items.extend(report.items);
            all_vectors.extend(vectors);
        }
    }

    let embedded_count = all_vectors.iter().filter(|v| !v.vector.is_empty()).count();
    let mut report = jis_router::report::BatchReport::new(all_items, all_vectors.len());
    report.sign(&jis_router::tibet::TibetFactory::new("chimera-cli"));

    let out_path = PathBuf::from("data/kmbit/batches").join(format!("{}.jsonl", report.batch_id));
    let _ = write_vectors_jsonl(&out_path, &all_vectors);

    println!("📦 Vectors written: {}", out_path.display());
    println!("   Total chunks: {}", all_vectors.len());
    if enable_embed {
        println!("   Embedded: {} (384 dimensions each)", embedded_count);
    }

    if print_report {
        let json = serde_json::to_string_pretty(&report).unwrap_or_default();
        println!("{}", json);
    }
}

fn run_diff(path: &Path, enable_embed: bool, gpu_index: usize, batch_size: usize) {
    // For now, diff is just a scan that reports changed files only.
    run_scan(path, true, enable_embed, gpu_index, batch_size);
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    // Supported extensions for indexing
    const EXTENSIONS: &[&str] = &["rs", "py", "md", "json", "toml", "txt", "js", "ts", "html", "yaml", "yml"];

    let mut files = Vec::new();
    if root.is_file() {
        if let Some(ext) = root.extension().and_then(|s| s.to_str()) {
            if EXTENSIONS.contains(&ext) {
                files.push(root.to_path_buf());
            }
        }
        return files;
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let p = path.to_string_lossy();
                // Skip heavy/irrelevant directories
                if p.contains("/.git") || p.contains("/target") ||
                   p.contains("/node_modules") || p.contains("/venv") ||
                   p.contains("/__pycache__") || p.contains("/.cache") ||
                   p.contains("/iso-builder") || p.contains("/chroot") ||
                   p.contains("/dist") || p.contains("/build") ||
                   p.contains("/.mypy_cache") || p.contains("/.pytest_cache") ||
                   p.contains("/vendor") || p.contains("/.tox") ||
                   p.contains("/coverage") || p.contains("/htmlcov") {
                    continue;
                }
                files.extend(collect_files(&path));
            } else {
                // Only collect files with supported extensions
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if EXTENSIONS.contains(&ext) {
                        files.push(path);
                    }
                }
            }
        }
    }
    files
}
