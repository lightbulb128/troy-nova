#include <vector>
#include "test.cuh"
#include "../src/ckks_encoder.cuh"
#include "test_adv.cuh"

using namespace troy;
using troy::utils::Array;
using troy::utils::ConstSlice;
using troy::utils::Slice;
using troy::utils::DynamicArray;
using std::vector;
using std::complex;
using tool::GeneralEncoder;
using tool::GeneralVector;
using tool::GeneralHeContext;

void ASSERT_TRUE(bool condition) {
    if (!condition) {
        printf("ASSERTION FAILED\n");
    }
}

void mainr(size_t n, bool inv) {
    GeneralHeContext gheh(false, SchemeType::BFV, n, 40, { 60, 60, 60 }, true, 0x123, 0);
    GeneralHeContext ghed( true, SchemeType::BFV, n, 40, { 60, 60, 60 }, true, 0x123, 0);
    
    uint64_t t = gheh.t();
    double scale = gheh.scale();
    double tolerance = gheh.tolerance();

    GeneralVector message = gheh.random_polynomial_full();

    Array<uint64_t> message_host = Array<uint64_t>::from_vector(vector(message.integers()));
    Array<uint64_t> message_device = message_host.to_device();

    if (inv) {
        utils::inverse_ntt_negacyclic_harvey(message_host.reference(), n, gheh.context()->first_context_data().value()->plain_ntt_tables());
        utils::inverse_ntt_negacyclic_harvey(message_device.reference(), n, ghed.context()->first_context_data().value()->plain_ntt_tables());
    } else {
        utils::ntt_negacyclic_harvey(message_host.reference(), n, gheh.context()->first_context_data().value()->plain_ntt_tables());
        utils::ntt_negacyclic_harvey(message_device.reference(), n, ghed.context()->first_context_data().value()->plain_ntt_tables());
    }

    Array<uint64_t> message_device_to_host = message_device.to_host();
    bool same = true;
    for (size_t i = 0; i < n; i++) {
        if (message_host[i] != message_device_to_host[i]) {
            same = false;
            break;
        }
    }
    std::cerr << "same: " << same << std::endl;

}

int main() {
    mainr(32, false);
    mainr(32, true);
    mainr(1024, false);
    mainr(1024, true);
    mainr(16384, false);
    mainr(16384, true);
    mainr(65536 * 2, false);
    mainr(65536 * 2, true);
    troy::utils::MemoryPool::Destroy();
}

// make custom && ./test/custom > a.log 2>&1