# How to solve the include errors in "interfaces"

You should first "make install" troy to somewhere, so that it produces the includes folder. You can specify where to install by "export DESTDIR=...". Then add to your intellisence this folder.
For clangd, you can add this line to your ".clangd" file:
```
CompileFlags:
  Add: 
    - -I/data/lxq/troy-nova/build/install/usr/local/include
```

# Philosophy

The cxx crate generates ".h" headers, so if we include any ".cuh" files, they cannot contain any kernel calls ("<<<>>>") directly, and all cuda related functions must be under a "#include<cuda_runtime.h>".
Furthermore, the cxx crate cannot automatically bind C++ static functions / constructors into Rust's impl methods without selves. So we first define all the FFI interfaces in "./reexports", capture them with "src/lib.rs" and finally encapsulate them as a Rust library.

# Workflow

1. Define in reexports.
2. Define in interfaces.rs.
3. Define a custom Rust encapsulator.
