//! # JIS Router - The Borrow Checker for Identity
//!
//! Just like Rust's borrow checker prevents memory bugs at compile time,
//! JIS Router prevents identity/trust violations at routing time.
//!
//! ## Core Concepts
//!
//! - **IDD** (Individual Device Derivate): Unique identity, not a number but an individual
//! - **Intent**: What you want to do (ERIN, ERAAN, EROMHEEN, ERACHTER)
//! - **FIR/A**: Trust score (0.0 = no trust, 1.0 = full trust)
//! - **Timeslot**: Time-gated access window
//! - **Lane**: A verified channel in AETHER
//!
//! ## The JIS Guarantee
//!
//! ```text
//! If an Intent compiles (passes JIS), it WILL execute safely.
//! No runtime trust violations. Correctness by construction.
//! ```
//!
//! One love, one fAmIly! 💙

pub mod types;
pub mod intent;
pub mod trust;
pub mod timeslot;
pub mod router;
pub mod tibet;
pub mod error;
pub mod space;
pub mod sema;
pub mod betti;
pub mod sentinel;
pub mod gfx;
pub mod snaft;
pub mod memory;
pub mod anchor;
pub mod refinery;
pub mod aindex;
pub mod chronos;
pub mod autonomy;
pub mod vision;
pub mod vault;
pub mod machtig;
pub mod negotiation;
pub mod tasks;
pub mod shield;
pub mod ingest;
pub mod discovery;
pub mod scanner;
pub mod briefing;
pub mod batch;
pub mod report;
pub mod vector;
pub mod kernel;
pub mod embedding;
pub mod oomllama;
pub mod oomllama_turbo;
pub mod quant;
pub mod gguf2oom;

pub use types::IDD;
pub use intent::{Intent, Action, IntentVerification};
pub use trust::*;
pub use timeslot::*;
pub use router::*;
pub use tibet::*;
pub use error::*;
pub use space::*;
pub use sema::{SemaRegistry, SemanticAddress, SemaResolution};
pub use betti::{BettiManager, ResourcePool, ResourceType, AllocationRequest, BettiStats, Allocation};
pub use sentinel::{SentinelClassifier, SentinelOutput};
pub use gfx::{GfxMonitor, GfxStatus, GpuInfo};
pub use snaft::{SnaftValidator, ThreatResult, ThreatType, SnaftStats};
pub use memory::{ConversationMemory, Conversation, ConversationMessage, MemoryStats};
pub use anchor::Anchor;
pub use refinery::{Refinery, PurityLevel, RefineResult};
pub use aindex::{AIndex, AIndexRecord};
pub use chronos::{ChronosManager, TimeCapsule, UnlockCondition, CapsuleState};
pub use autonomy::AutonomyDaemon;
pub use vision::{VisionRouter, VisionProvider, VisionPreference};
pub use vault::TibetVault;
pub use machtig::{Machtiging, Constraint};
pub use negotiation::{Offer, Agreement, NegotiationContext};
pub use tasks::{Task, TaskStatus, TaskType};
pub use discovery::{Discovery, DiscoveryRadar, DiscoveryType};
pub use scanner::ChimeraScanner;
pub use briefing::{MorningBriefing, BriefingEngine};
pub use shield::{LiabilityShield, SettlementAction};
pub use ingest::{IngestManager, IngestJob, IngestStatus};
pub use batch::{BatchProcessor, BatchConfig};
pub use report::{BatchReport, BatchItemReport};
pub use vector::{VectorRecord, VectorMeta};
pub use kernel::{SovereignKernel, SovereignIdentity, NeuralCoreInfo};
pub use embedding::EmbeddingEngine;
pub use oomllama::OomLlama;
pub use oomllama_turbo::{
    TurboEngine, TurboConfig, ModelKVCache, LayerPin, PinStrategy,
    FlashAttentionConfig, flash_attention_forward,
    compute_rope_freqs, apply_rope,
};


/// JIS Router version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The heart of HumoticaOS - Jasper's heartbeat interval
pub const HEARTBEAT_INTERVAL_MS: u64 = 1000;
// Triggering GPU indexing at za 10 jan 2026 14:15:51 CET
pub mod oom_inference;

// Python bindings (only when feature enabled)
#[cfg(feature = "python")]
pub mod python;
