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

## 2024-06-29

- Allow user to create `MemoryPoolHandle`s and supply them to API calls.
    - Add multithread tests and benchmark.
    - Update pybind11 encapsulation and test folder structure.

## 2024-07-03

- Move the implementation of MemoryPool to different input files. Directly using macros in the header may lead to user's on including troy wrong with defines not given.

## 2024-07-14

- Allow scaled up polynomial plaintexts to store only partial coefficients, to save device memory when only a small degree polynomial is encoded.

## 2024-08-09

- Update kernels for multiple APIs in evaluator and encryptor, removing redundant cuda memory copies and memsets.
- All unittests using device will be skipped if no device is detected on the machine.
- Examples will run on host if no device is detected on the machine.

## 2024-09-06

- Use `cudaMemcpyAsync` instead of `cudaMemcpy` when copying from host to device, and device to device.
- Use `cudaMemsetAsync` instead of `cudaMemset` when setting device memory.
