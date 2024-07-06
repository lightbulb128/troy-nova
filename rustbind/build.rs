fn main() {

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=reexports");
    
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
    let destination = cmake_config.build();
    println!("destination: {:?}", destination.display());

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // link static library of troy
    println!("cargo:rustc-link-search=native={}", out_dir.clone() + "/lib");

    cxx_build::bridge("src/interfaces.rs")
        .cuda(true)
        .flag("-std=c++17")
        .flag("-Xcompiler")
        .flag("-w")
        .include(out_dir.clone() + "/include")
        .cpp_link_stdlib("troy")
        .compile("rustbind");

    
}