//! With the `gpu` feature, compile cuda/river.cu with nvcc into a static
//! library and link it with the CUDA runtime.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/river.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("river.o");
    let ok = Command::new(format!("{cuda}/bin/nvcc"))
        .args(["-O3", "-std=c++17", "-gencode", "arch=compute_121,code=[sm_121,compute_121]", "-fmad=false", "-Xcompiler", "-fPIC", "-c", "cuda/river.cu", "-o"])
        .arg(&obj)
        .status()
        .expect("running nvcc (set CUDA_HOME, or build without the gpu feature)")
        .success();
    assert!(ok, "nvcc failed on cuda/river.cu");
    let lib = out.join("libhugpu.a");
    let _ = std::fs::remove_file(&lib);
    assert!(Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().expect("running ar").success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=hugpu");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=cudart_static");
    for l in ["stdc++", "dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
}
