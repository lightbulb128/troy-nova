from test import GeneralVectorDataType, GeneralEncoder, GeneralVector, GeneralHeContext
import pytroy
from pytroy import SchemeType
import numpy as np
import unittest
import argparse

class Heconv2dTest:

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
        
        x_encoded = helper.encode_inputs(encoder.encoder, x.data)
        w_encoded = helper.encode_weights(encoder.encoder, w.data)
        s_encoded = helper.encode_outputs(encoder.encoder, s.data)

        x_encrypted = x_encoded.encrypt_symmetric(encryptor)
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


def create_test_class(ghe: GeneralHeContext):

    class UnnamedClass(unittest.TestCase):

        def setUp(self) -> None:
            self.ghe = ghe
            self.tester = Heconv2dTest(self.ghe, self)
            return super().setUp()

    if ghe.scheme == SchemeType.BFV:
        UnnamedClass.test_conv2d_small_no_pack = lambda self: self.tester.test_conv2d(2, 3, 6, 7, 9, 3, 5, mod_switch_to_next=False)
        UnnamedClass.test_conv2d_medium_no_pack = lambda self: self.tester.test_conv2d(2, 3, 10, 56, 56, 10, 10, mod_switch_to_next=False)

    return UnnamedClass

class HostTestSuite(unittest.TestSuite):

    def __init__(self):
        super().__init__()
        test_case = create_test_class(GeneralHeContext(False, SchemeType.BFV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))

class DeviceTestSuite(unittest.TestSuite):

    def __init__(self):
        super().__init__()
        test_case = create_test_class(GeneralHeContext(True, SchemeType.BFV, 8192, 20, [60, 40, 40, 60], True, 0x123))
        self.addTest(unittest.makeSuite(test_case))


def custom_main():
    print("There is nothing here.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", action="store_true")

    args = parser.parse_args()

    pytroy.initialize_kernel(0)

    if args.custom:

        custom_main()

    else:

        # run host suite
        print("Running host test suite")
        suite = HostTestSuite()
        unittest.TextTestRunner().run(suite)
        print("")

        # run device suite
        print("Running device test suite")
        suite = DeviceTestSuite()
        unittest.TextTestRunner().run(suite)
        print("")

    pytroy.destroy_memory_pool()