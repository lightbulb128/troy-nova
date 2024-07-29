import pytroy

if __name__ == "__main__":

    params = pytroy.EncryptionParameters(pytroy.SchemeType.BFV)
    params.set_poly_modulus_degree(32)
    params.set_plain_modulus(pytroy.PlainModulus.batching(32, 30))
    params.set_coeff_modulus(pytroy.CoeffModulus.create(32, [40, 40, 40]))

    context = pytroy.HeContext(params, sec_level=pytroy.SecurityLevel.Nil)
    
    encoder = pytroy.BatchEncoder(context)
    values = [1, 2, 3, 4]
    plain = encoder.encode_simd_new(values)
    print(plain.obtain_data())

    decoded = encoder.decode_simd_new(plain)
    print(decoded)

