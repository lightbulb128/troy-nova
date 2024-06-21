## 2024-06-06

Use NTT form for BGV Ciphertexts. (Microsoft SEAL 4.1.0)

## 2024-06-12

Use AES-CTR (inspired by torchcsprng) to generate randomness. In the previous versions, we used the cuRAND/BLAKE2 to generate randomness for GPU/CPU respectively. But cuRAND's default XORWOW is not cryptographic secure.

## 2024-06-20

Implement CKKS matrix multiplication.

## 2024-06-21

- Add examples.
- Provide an `unsafe` implementation of the Memory Pool.