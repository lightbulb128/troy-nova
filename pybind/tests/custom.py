import pytroy
import numpy as np

if __name__ == "__main__":

    params = pytroy.EncryptionParameters(pytroy.SchemeType.BFV)
    params.set_poly_modulus_degree(32)
    params.set_plain_modulus(pytroy.PlainModulus.batching(32, 30))
    params.set_coeff_modulus(pytroy.CoeffModulus.create(32, [40, 40, 40]))

    context = pytroy.HeContext(params, sec_level=pytroy.SecurityLevel.Nil)
    
    encoder = pytroy.BatchEncoder(context)
    values = [1, 2, 3, 4]
    plain = encoder.encode_simd_new(values)
    data = plain.obtain_data()
    assert(data[0] == 738197241)

    decoded = encoder.decode_simd_new(plain)
    assert(np.sum(decoded[:4] == values) == 4)

