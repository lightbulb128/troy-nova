from utils import GeneralVectorDataType, GeneralEncoder, GeneralVector, GeneralHeContext
import pytroy
from pytroy import SchemeType
import numpy as np
import unittest
import argparse

class HeUint64Conv2dTest:

    def __init__(self, ghe: GeneralHeContext, tester: unittest.TestCase):
        self.ghe = ghe
        self.tester = tester

    def test_conv2d(self, 
        bs: int, ic: int, oc: int, ih: int, iw: int, kh: int, kw: int,
        mod_switch_to_next: bool
    ):
        ghe = self.ghe
        if ghe.scheme != SchemeType.BFV: return
        t = ghe.t
        def mulmod(a, b):
            return (int(a) * int(b)) % t
        def addmod(a, b):
            return (int(a) + int(b)) % t
        
        x = ghe.random_polynomial(bs * ic * ih * iw)
        w = ghe.random_polynomial(oc * ic * kh * kw)
        oh = ih - kh + 1
        ow = iw - kw + 1
        s = ghe.random_polynomial(bs * oc * oh * ow)
        helper = pytroy.Conv2dHelper(bs, ic, oc, ih, iw, kh, kw, 
            ghe.parms.poly_modulus_degree(), pytroy.MatmulObjective.EncryptLeft)
        
        he = ghe.context
        encoder = ghe.encoder
        encryptor = ghe.encryptor
        decryptor = ghe.decryptor
        evaluator = ghe.evaluator
        
        x_encrypted = helper.encrypt_inputs(encryptor, encoder.encoder, x.data)
        w_encoded = helper.encode_weights(encoder.encoder, w.data)
        s_encoded = helper.encode_outputs(encoder.encoder, s.data)

        x_serialized = x_encrypted.save(he)
        x_encrypted = pytroy.Cipher2d.load_new(x_serialized, he)

        y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded)
        if mod_switch_to_next:
            y_encrypted.mod_switch_to_next_inplace(evaluator)
        y_encrypted.add_plain_inplace(evaluator, s_encoded)

        y_serialized = helper.serialize_outputs(evaluator, y_encrypted)
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized)

        y_decrypted = helper.decrypt_outputs(encoder.encoder, decryptor, y_encrypted)

        x_np_mat = np.reshape(x.data, (bs, ic, ih, iw))
        w_np_mat = np.reshape(w.data, (oc, ic, kh, kw))
        s_np_mat = np.reshape(s.data, (bs, oc, oh, ow))
        y_np_mat = np.zeros((bs, oc, oh, ow), dtype=np.uint64)
        for b in range(bs):
            for o in range(oc):
                for i in range(ic):
                    for ki in range(kh):
                        for kj in range(kw):
                            y_np_mat[b, o, :, :] += x_np_mat[b, i, ki:ki+oh, kj:kj+ow] * w_np_mat[o, i, ki, kj]
        y_np_mat += s_np_mat
        y_truth = np.reshape(y_np_mat, (bs * oc * oh * ow,)) % t

        decrypted = GeneralVector(GeneralVectorDataType.Integers, y_decrypted)
        truth = GeneralVector(GeneralVectorDataType.Integers, y_truth)
        self.tester.assertTrue(ghe.near_equal(decrypted, truth))


def create_test_uint64s_class(name, ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = HeUint64Conv2dTest(self.ghe, self)
            return super().setUp()

    if ghe.scheme == SchemeType.BFV or ghe.scheme == SchemeType.BGV:
        UnnamedClass.test_conv2d_small_no_pack = lambda self: self.tester.test_conv2d(2, 3, 6, 7, 9, 3, 5, mod_switch_to_next=False)
        UnnamedClass.test_conv2d_medium_no_pack = lambda self: self.tester.test_conv2d(2, 3, 10, 56, 56, 10, 10, mod_switch_to_next=False)

    return type(name, (UnnamedClass,), {})

class HeDoubleConv2dTest:

    def __init__(self, ghe: GeneralHeContext, tester: unittest.TestCase):
        self.ghe = ghe
        self.tester = tester

    def test_conv2d(self, 
        bs: int, ic: int, oc: int, ih: int, iw: int, kh: int, kw: int,
        mod_switch_to_next: bool
    ):
        ghe = self.ghe
        if ghe.scheme != SchemeType.CKKS: return
        
        x = ghe.random_polynomial(bs * ic * ih * iw)
        w = ghe.random_polynomial(oc * ic * kh * kw)
        oh = ih - kh + 1
        ow = iw - kw + 1
        s = ghe.random_polynomial(bs * oc * oh * ow)
        helper = pytroy.Conv2dHelper(bs, ic, oc, ih, iw, kh, kw, 
            ghe.parms.poly_modulus_degree(), pytroy.MatmulObjective.EncryptLeft)
        
        he = ghe.context
        encoder = ghe.encoder
        encryptor = ghe.encryptor
        decryptor = ghe.decryptor
        evaluator = ghe.evaluator
        
        x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder.encoder, x.data, None, ghe.scale)
        w_encoded = helper.encode_weights_doubles(encoder.encoder, w.data, None, ghe.scale)
        s_encoded = helper.encode_outputs_doubles(encoder.encoder, s.data, None, ghe.scale * ghe.scale)

        x_serialized = x_encrypted.save(he)
        x_encrypted = pytroy.Cipher2d.load_new(x_serialized, he)

        y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded)
        if mod_switch_to_next:
            y_encrypted.mod_switch_to_next_inplace(evaluator)
        y_encrypted.add_plain_inplace(evaluator, s_encoded)

        y_serialized = helper.serialize_outputs(evaluator, y_encrypted)
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized)

        y_decrypted = helper.decrypt_outputs_doubles(encoder.encoder, decryptor, y_encrypted)

        x_np_mat = np.reshape(x.data, (bs, ic, ih, iw))
        w_np_mat = np.reshape(w.data, (oc, ic, kh, kw))
        s_np_mat = np.reshape(s.data, (bs, oc, oh, ow))
        y_np_mat = np.zeros((bs, oc, oh, ow), dtype=np.double)
        for b in range(bs):
            for o in range(oc):
                for i in range(ic):
                    for ki in range(kh):
                        for kj in range(kw):
                            y_np_mat[b, o, :, :] += x_np_mat[b, i, ki:ki+oh, kj:kj+ow] * w_np_mat[o, i, ki, kj]
        y_np_mat += s_np_mat
        y_truth = np.reshape(y_np_mat, (bs * oc * oh * ow,))

        decrypted = GeneralVector(GeneralVectorDataType.Doubles, y_decrypted)
        truth = GeneralVector(GeneralVectorDataType.Doubles, y_truth)
        self.tester.assertTrue(ghe.near_equal(decrypted, truth))


def create_test_doubles_class(name, ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = HeDoubleConv2dTest(self.ghe, self)
            return super().setUp()

    UnnamedClass.test_conv2d_small_no_pack = lambda self: self.tester.test_conv2d(2, 3, 6, 7, 9, 3, 5, mod_switch_to_next=False)
    UnnamedClass.test_conv2d_medium_no_pack = lambda self: self.tester.test_conv2d(2, 3, 10, 56, 56, 10, 10, mod_switch_to_next=False)

    return type(name, (UnnamedClass,), {})

class HeRing2kConv2dTest:

    def __init__(self, device: bool, t_bits: int, poly_degree: int, q_bits: "list[int]", tester: unittest.TestCase):
        self.tester = tester
        params = pytroy.EncryptionParameters(pytroy.SchemeType.BFV)
        params.set_poly_modulus_degree(poly_degree)
        params.set_plain_modulus(1 << 20) # this is not important
        params.set_coeff_modulus(pytroy.CoeffModulus.create(poly_degree, q_bits))
        t_mask = (1 << t_bits) - 1

        context = pytroy.HeContext(params, True, pytroy.SecurityLevel.Nil, 0x123)
        if t_bits > 32:
            encoder = pytroy.PolynomialEncoderRing2k64(context, t_bits)
        else:
            encoder = pytroy.PolynomialEncoderRing2k32(context, t_bits)
        
        if device:
            context.to_device_inplace()
            encoder.to_device_inplace()
        
        keygen = pytroy.KeyGenerator(context)
        encryptor = pytroy.Encryptor(context)
        encryptor.set_secret_key(keygen.secret_key())
        decryptor = pytroy.Decryptor(context, keygen.secret_key())
        evaluator = pytroy.Evaluator(context)
        automorphism_keys = keygen.create_automorphism_keys(False)

        self.poly_degree = poly_degree
        self.context = context
        self.encoder = encoder
        self.encryptor = encryptor
        self.decryptor = decryptor
        self.evaluator = evaluator
        self.automorphism_keys = automorphism_keys
        self.t_bits = t_bits
        self.t_mask = t_mask

    def random_polynomial(self, n: int):
        if self.t_bits > 32:
            return np.random.randint(0, np.iinfo(np.uint64).max, n, dtype=np.uint64) & self.t_mask
        else:
            return np.random.randint(0, np.iinfo(np.uint32).max, n, dtype=np.uint32) & self.t_mask

    def test_conv2d(self, 
        bs: int, ic: int, oc: int, ih: int, iw: int, kh: int, kw: int,
        mod_switch_to_next: bool
    ):
        
        x = self.random_polynomial(bs * ic * ih * iw)
        w = self.random_polynomial(oc * ic * kh * kw)
        oh = ih - kh + 1
        ow = iw - kw + 1
        s = self.random_polynomial(bs * oc * oh * ow)
        helper = pytroy.Conv2dHelper(bs, ic, oc, ih, iw, kh, kw, 
            self.poly_degree, pytroy.MatmulObjective.EncryptLeft)
        
        he = self.context
        encoder = self.encoder
        encryptor = self.encryptor
        decryptor = self.decryptor
        evaluator = self.evaluator
        
        if self.t_bits > 32:
            x_encrypted = helper.encrypt_inputs_ring2k64(encryptor, encoder, x.data, None)
            w_encoded = helper.encode_weights_ring2k64(encoder, w.data, None)
            s_encoded = helper.encode_outputs_ring2k64(encoder, s.data, None)
        else:
            x_encrypted = helper.encrypt_inputs_ring2k32(encryptor, encoder, x.data, None)
            w_encoded = helper.encode_weights_ring2k32(encoder, w.data, None)
            s_encoded = helper.encode_outputs_ring2k32(encoder, s.data, None)

        x_serialized = x_encrypted.save(he)
        x_encrypted = pytroy.Cipher2d.load_new(x_serialized, he)

        y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded)
        if mod_switch_to_next:
            y_encrypted.mod_switch_to_next_inplace(evaluator)
        y_encrypted.add_plain_inplace(evaluator, s_encoded)

        y_serialized = helper.serialize_outputs(evaluator, y_encrypted)
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized)

        if self.t_bits > 32:
            y_decrypted = helper.decrypt_outputs_ring2k64(encoder, decryptor, y_encrypted)
        else:
            y_decrypted = helper.decrypt_outputs_ring2k32(encoder, decryptor, y_encrypted)

        x_np_mat = np.reshape(x.data, (bs, ic, ih, iw))
        w_np_mat = np.reshape(w.data, (oc, ic, kh, kw))
        s_np_mat = np.reshape(s.data, (bs, oc, oh, ow))
        y_np_mat = np.zeros((bs, oc, oh, ow), dtype=np.uint64)
        for b in range(bs):
            for o in range(oc):
                for i in range(ic):
                    for ki in range(kh):
                        for kj in range(kw):
                            y_np_mat[b, o, :, :] += x_np_mat[b, i, ki:ki+oh, kj:kj+ow] * w_np_mat[o, i, ki, kj]
        y_np_mat += s_np_mat
        y_truth = np.reshape(y_np_mat, (bs * oc * oh * ow,)) & self.t_mask

        self.tester.assertTrue(np.allclose(y_decrypted, y_truth))

def create_test_ring2k_class(name, device: bool, t_bits: int, poly_degree: int, q_bits: "list[int]"):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.tester = HeRing2kConv2dTest(device, t_bits, poly_degree, q_bits, self)
            return super().setUp()

    UnnamedClass.test_conv2d_small_no_pack = lambda self: self.tester.test_conv2d(2, 3, 6, 7, 9, 3, 5, mod_switch_to_next=False)
    UnnamedClass.test_conv2d_medium_no_pack = lambda self: self.tester.test_conv2d(2, 3, 10, 56, 56, 10, 10, mod_switch_to_next=False)

    return type(name, (UnnamedClass,), {})


class CompleteTestSuite(unittest.TestSuite):

    def __init__(self, device):
        super().__init__()
        devstr = "Device" if device else "Host"
        
        test_case = create_test_uint64s_class(devstr + "BFVConv2d", GeneralHeContext(device, SchemeType.BFV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_uint64s_class(devstr + "BGVConv2d", GeneralHeContext(device, SchemeType.BGV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))
        
        test_case = create_test_doubles_class(devstr + "CKKSConv2d", GeneralHeContext(device, SchemeType.CKKS, 8192, 20, [60, 40, 40, 60], True, 0x123, 5, 1<<20, 1e-2))
        self.addTest(unittest.makeSuite(test_case))
        
        test_case = create_test_ring2k_class(devstr + "Ring32Conv2d", device, 32, 8192, [60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring20Conv2d", device, 20, 8192, [60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring17Conv2d", device, 17, 8192, [60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))

        test_case = create_test_ring2k_class(devstr + "Ring64Conv2d", device, 64, 8192, [60, 60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring50Conv2d", device, 50, 8192, [60, 60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))
        test_case = create_test_ring2k_class(devstr + "Ring33Conv2d", device, 33, 8192, [60, 60, 60, 60])
        self.addTest(unittest.makeSuite(test_case))

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(CompleteTestSuite(False))
    suite.addTest(CompleteTestSuite(True))
    return suite

if __name__ == "__main__":
    pytroy.initialize_kernel(0)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
    pytroy.destroy_memory_pool()