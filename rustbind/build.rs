use std::env;
use std::path::PathBuf;
use std::path::Path;

// ensure that exactly one of "memory-pool-none", "memory-pool-unsafe" or "memory-pool-safe" is set
#[cfg(all(feature = "memory-pool-none", all(feature = "memory-pool-unsafe")))]
compile_error!("memory-pool-none and memory-pool-unsafe are mutually exclusive");

#[cfg(all(feature = "memory-pool-none", all(feature = "memory-pool-safe")))]
compile_error!("memory-pool-none and memory-pool-safe are mutually exclusive");

#[cfg(all(feature = "memory-pool-unsafe", all(feature = "memory-pool-safe")))]
compile_error!("memory-pool-unsafe and memory-pool-safe are mutually exclusive");

#[cfg(not(any(feature = "memory-pool-none", feature = "memory-pool-unsafe", feature = "memory-pool-safe")))]
compile_error!("Exactly one of memory-pool-none, memory-pool-unsafe or memory-pool-safe must be set");


fn main() {

    // check dependencies
    // -- check cuda exist
    let cuda_include = "/usr/local/cuda/include";
    if !Path::new(cuda_include).exists() {
        panic!("CUDA not found in /usr/local/cuda");
    }
    // -- check cuda_runtime.h exist
    let cuda_runtime = "/usr/local/cuda/include/cuda_runtime.h";
    if !Path::new(cuda_runtime).exists() {
        panic!("cuda_runtime.h not found in /usr/local/cuda/include");
    }

    // rerun if changed
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-changed=wrapper.h");

    
    // compile troy
    let mut cmake_config = cmake::Config::new("../");
    #[cfg(feature = "memory-pool-none")] {
        cmake_config.define("TROY_MEMORY_POOL", "OFF");
    }
    #[cfg(feature = "memory-pool-unsafe")] {
        cmake_config.define("TROY_MEMORY_POOL", "ON");
        cmake_config.define("TROY_MEMORY_POOL_UNSAFE", "ON");
    }
    #[cfg(feature = "memory-pool-safe")] {
        cmake_config.define("TROY_MEMORY_POOL", "ON");
        cmake_config.define("TROY_MEMORY_POOL_UNSAFE", "OFF");
    }
    cmake_config.define("TROY_PYBIND", "OFF");
    cmake_config.define("TROY_TEST", "OFF");
    // -- we need to keep the inline functions for gcc
    cmake_config.define("CMAKE_CXX_FLAGS", "-fkeep-inline-functions");
    cmake_config.define("CMAKE_CUDA_FLAGS", "-Xcompiler -fkeep-inline-functions");
    let destination = cmake_config.build();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // build wrapper.cpp with cc
    cc::Build::new()
        .cpp(true)
        .file("wrapper.cpp")
        .include(out_dir.clone() + "/include")
        .include(cuda_include)
        // compress all warnings
        .warnings(false)
        .cpp_link_stdlib("stdc++")
        .cpp_link_stdlib("cudart")
        .cpp_link_stdlib("cuda")
        .cpp_link_stdlib("troy_static")
        .out_dir(out_dir.clone() + "/lib")
        .compile("wrapper");


    // link static library of troy
    println!("cargo:rustc-link-search=native={}", out_dir.clone() + "/lib");
    println!("cargo:rustc-link-lib=static=troy_static");

    // link cuda
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // link stdc++
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // link wrapper static
    println!("cargo:rustc-link-search=native={}", out_dir.clone() + "/lib");
    println!("cargo:rustc-link-lib=static=wrapper");

    let bindings = bindgen::Builder::default()

        // cuda include
        .clang_arg(format!("-I{}", cuda_include))

        // troy include
        .clang_arg(format!("-I{}", out_dir.clone() + "/include"))
        .clang_args(&["-x", "c++"])
        .clang_args(&["-std=c++17"])

        // compress all warnings
        .clang_args(&["-w"])
        
        .generate_inline_functions(true)
        
        .header("wrapper.h")
        // .allowlist_type("troy::utils::MemoryPool")
        .opaque_type("std::.*")
        .allowlist_function("troy_wrapper::.*")
        .allowlist_type("troy::utils::MemoryPoolHandle")
        .allowlist_type("troy::utils::MemoryPool")

        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))

        // Finish the builder and generate the bindings.
        .generate()

        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}