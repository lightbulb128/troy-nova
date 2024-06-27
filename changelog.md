## 2024-06-06

- Use NTT form for BGV Ciphertexts. (Microsoft SEAL 4.1.0)

## 2024-06-12

- Use AES-CTR (inspired by torchcsprng) to generate randomness. In the previous versions, we used the cuRAND/BLAKE2 to generate randomness for GPU/CPU respectively. But cuRAND's default XORWOW is not cryptographic secure.

## 2024-06-20

- Implement CKKS matrix multiplication.

## 2024-06-21

- Add examples.
- Provide an `unsafe` implementation of the Memory Pool.

## 2024-06-24

- Implement ciphertext-plaintext support for Ring2k BFV.
    - The user may use plain modulus $t = 2^k$ for $k \leq 128$, with `PolynomialEncoderRing2k<T>`, where `T = uint32_t, uint64_t, uint128_t`, and the given `k` must be greater than half of the type `T`'s bitwidth.
    - `scale_up`, `centralize` are used for encoding, while `scale_down` is for decoding. See the evaluator unit tests for some usages.
    - Matrix multiplication support for Ring2k-BFV.
    - Pybind11 encapsulation. This only includes `uint32_t` and `uint64_t` versions, since native 128-bit support is missing in pybind11/python/numpy.

## 2024-06-26

- Implement `invariant_noist_budget` in `Decryptor`, available for BFV and BGV.

## 2024-06-27

- Allow user to create `MemoryPoolHandle`s and supply them to API calls.