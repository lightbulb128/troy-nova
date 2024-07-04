fn main() {

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=interfaces");
    
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

    cxx_build::bridge("src/lib.rs")
        .cuda(true)
        .include(out_dir + "/include")
        .file("interfaces/memory_pool.cu")
        .compile("rustbind");
    
}