#include <vector>
#include "test.cuh"
#include "../src/he_context.cuh"

using namespace troy;
using troy::utils::Array;
using std::vector;

int main() {
    
    SchemeType scheme = SchemeType::BFV;
    EncryptionParameters parms(scheme);

    auto context = HeContext::create(parms, false, SecurityLevel::None);

    return 0;
}