#[cxx::bridge]
pub(crate) mod ffi {

    // utility functions
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/reexports/basics.h");
        fn device_count() -> usize;
    }

    // scheme type
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/reexports/basics.h");
        type SchemeType = crate::basics::SchemeType;
        fn scheme_type_to_string(scheme_type: SchemeType) -> String;
    }

    // security level
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/reexports/basics.h");
        type SecurityLevel = crate::basics::SecurityLevel;
        fn security_level_to_string(scheme_type: SecurityLevel) -> String;
    }

    // memory pool
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/reexports/memory_pool.h");
        type MemoryPool;
        fn handle_address(self: &MemoryPool) -> usize;
        fn device_index(self: &MemoryPool) -> usize;
        fn memory_pool_constructor(device_index: usize) -> UniquePtr<MemoryPool>;
        fn memory_pool_constructor_copy(other: &MemoryPool) -> UniquePtr<MemoryPool>;
        fn memory_pool_static_global_pool() -> UniquePtr<MemoryPool>;
        fn memory_pool_static_destroy();
        fn memory_pool_static_nullptr() -> UniquePtr<MemoryPool>;
    }

    // parms id
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/reexports/encryption_parameters.h");
        type ParmsID;
        fn parms_id_constructor_copy(other: &ParmsID) -> UniquePtr<ParmsID>;
        fn parms_id_static_zero() -> UniquePtr<ParmsID>;
        fn equals_to(self: &ParmsID, other: &ParmsID) -> bool;
        fn to_array(self: &ParmsID) -> [u64; 4];
        fn is_zero(self: &ParmsID) -> bool;
    }

    // modulus
    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("troy-rustbind/reexports/modulus.h");
        type Modulus;
        fn modulus_constructor(value: u64) -> UniquePtr<Modulus>;
        fn modulus_constructor_copy(other: &Modulus) -> UniquePtr<Modulus>;
        fn is_prime(self: &Modulus) -> bool;
        fn is_zero(self: &Modulus) -> bool;
        fn value(self: &Modulus) -> u64;
        fn bit_count(self: &Modulus) -> usize;
        fn to_string(self: &Modulus) -> String;
        fn reduce(self: &Modulus, value: u64) -> u64;
        fn reduce_u128(self: &Modulus, high: u64, low: u64) -> u64;
        fn reduce_mul_u64(self: &Modulus, operand1: u64, operand2: u64) -> u64;
        // statics
        fn coeff_modulus_max_bit_count(poly_modulus_degree: usize, security_level: SecurityLevel) -> usize;
        fn coeff_modulus_bfv_default(poly_modulus_degree: usize, sec: SecurityLevel, output: &mut Vec<UniquePtr<Modulus>>);

    }

    #[namespace = "troy_rust"]
    unsafe extern "C++" {
        include!("foo.h");
        type Foo;
        fn create_foo() -> UniquePtr<Foo>;
        fn create_foos(count: usize) -> CxxVector<UniquePtr<Foo>>;
    }

}
