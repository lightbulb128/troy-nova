#include "../../src/utils/ntt.h"
#include "../../src/coeff_modulus.h"
#include "../../src/utils/timer.h"

using troy::utils::NTTTables;
using troy::utils::Box;
using troy::utils::Array;
using troy::Modulus;
using troy::PlainModulus;
using troy::bench::Timer;

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Usage: <executable-name> <logn> <logt> [<repeat-times>=10]" << std::endl;
        return 0;
    }

    int logn = atoi(argv[1]);
    int logt = atoi(argv[2]);

    int repeat = 10;
    if (argc >= 4) {
        repeat = atoi(argv[3]);
    }

    size_t n = 1 << logn;
    Modulus t = PlainModulus::batching(n, logt);

    Box<NTTTables> table(new NTTTables(logn, t), false);
    table->to_device_inplace();
    table.to_device_inplace();

    Array<uint64_t> a(n, false);
    for (size_t i = 0; i < n; i++) {
        a[i] = t.reduce(rand());
    }
    a.to_device_inplace();

    Timer timer;
    size_t timer_ntt = timer.register_timer("NTT");
    size_t timer_intt = timer.register_timer("INTT");

    for (size_t i = 0; i < repeat; i++) {
    
        timer.tick(timer_ntt);
        troy::utils::ntt_negacyclic_harvey(a.reference(), n, table.as_const_pointer());
        timer.tock(timer_ntt);

        timer.tick(timer_intt);
        troy::utils::inverse_ntt_negacyclic_harvey(a.reference(), n, table.as_const_pointer());
        timer.tock(timer_intt);

    }

    timer.print_divided(repeat);

    troy::utils::MemoryPool::Destroy();

    return 0;
}