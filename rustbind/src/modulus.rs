use super::ffi;
use crate::SecurityLevel;

type UpModulus = cxx::UniquePtr<ffi::Modulus>;

pub struct Modulus {
    p: UpModulus,
}

impl Modulus {

    #[inline]
    pub fn new(value: u64) -> Self {
        let p = ffi::modulus_constructor(value);
        Self { p }
    }

    #[inline]
    pub fn is_prime(&self) -> bool {
        self.p.is_prime()
    }

}

pub struct CoeffModulus {}

impl CoeffModulus {

    // fn bfv_default(poly_modulus_degree: usize, sec: SecurityLevel) -> Vec<Modulus> {
    //     let mut cxx_result: cxx::UniquePtr<cxx::CxxVector<crate::ffi::Modulus>> = cxx::CxxVector::new();
    //     let pinned = std::pin::Pin::new(&mut cxx_result);
    //     ffi::coeff_modulus_bfv_default(poly_modulus_degree, sec, pinned);
    //     cxx_result.into_iter().map(|m| Modulus { p: cxx::UniquePtr::new(m) }).collect()
    // }

}