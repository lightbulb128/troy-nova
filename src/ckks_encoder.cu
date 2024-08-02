#include "ckks_encoder.h"

namespace troy {

    using utils::Array;
    using utils::ConstSlice;
    using utils::Slice;
    using utils::ConstPointer;
    using utils::Pointer;
    using std::complex;

    struct CustomComplex {
        double real;
        double imag;

        __host__ __device__
        inline CustomComplex() : real(0), imag(0) {}

        __host__ __device__
        inline CustomComplex(double real, double imag) : real(real), imag(imag) {}

        __host__ __device__
        inline CustomComplex neg() {
            return CustomComplex(-this->real, -this->imag);
        }

        __host__ __device__
        inline CustomComplex conj() {
            return CustomComplex(this->real, -this->imag);
        }

        __host__ __device__
        inline CustomComplex add(CustomComplex other) {
            return CustomComplex(this->real + other.real, this->imag + other.imag);
        }

        __host__ __device__
        inline CustomComplex sub(CustomComplex other) {
            return CustomComplex(this->real - other.real, this->imag - other.imag);
        }

        __host__ __device__
        inline CustomComplex mul(CustomComplex other) {
            return CustomComplex(
                this->real * other.real - this->imag * other.imag, 
                this->real * other.imag + this->imag * other.real
            );
        }

        __host__ __device__
        inline CustomComplex mul(double other) {
            return CustomComplex(this->real * other, this->imag * other);
        }

        static ConstSlice<CustomComplex> slice(ConstSlice<complex<double>> r) {
            return ConstSlice<CustomComplex>(
                reinterpret_cast<const CustomComplex*>(r.raw_pointer()), r.size(), r.on_device(), r.pool()
            );
        }

        static Slice<CustomComplex> slice(Slice<complex<double>> r) {
            return Slice<CustomComplex>(
                reinterpret_cast<CustomComplex*>(r.raw_pointer()), r.size(), r.on_device(), r.pool()
            );
        }
    };

    static complex<double> mirror(complex<double> root) {
        return complex<double>(root.imag(), root.real());
    }

    static complex<double> conj(complex<double> root) {
        return complex<double>(root.real(), -root.imag());
    }

    /* unused 
    __host__ __device__
    static CustomComplex mirror(CustomComplex root) {
        return CustomComplex(root.imag, root.real);
    }
    */

    __device__
    static CustomComplex conj(CustomComplex root) {
        return CustomComplex(root.real, -root.imag);
    }

    struct ComplexRoots {
        std::vector<complex<double>> roots;
        size_t degree_of_roots;

        ComplexRoots(size_t degree_of_roots) {
            // Generate 1/8 of all roots.
            // Alternatively, choose from precomputed high-precision roots in files.
            size_t n = degree_of_roots / 8;
            this->roots.reserve(n + 1);
            for (size_t i = 0; i <= n; i++) {
                double theta = 2 * M_PI * static_cast<double>(i) / static_cast<double>(degree_of_roots);
                this->roots.push_back(complex<double>(std::cos(theta), std::sin(theta)));
            }
            this->degree_of_roots = degree_of_roots;
        }

        complex<double> get_root(size_t index) {
            size_t degree_of_roots = this->degree_of_roots;
            index &= (degree_of_roots - 1);
            if (index <= degree_of_roots / 8) {
                return this->roots[index];
            } else if (index <= degree_of_roots / 4) {
                return mirror(this->roots[degree_of_roots / 4 - index]);
            } else if (index < degree_of_roots / 2) {
                return -conj(get_root(degree_of_roots / 2 - index));
            } else if (index <= 3 * degree_of_roots / 4) {
                return -get_root(index - degree_of_roots / 2);
            } else {
                return conj(get_root(degree_of_roots - index));
            }
        }
    };

    CKKSEncoder::CKKSEncoder(HeContextPointer context) {
        
        if (context->on_device()) {
            throw std::invalid_argument("[CKKSEncoder::CKKSEncoder] Cannot create from device context.");
        }
        if (!context->parameters_set()) {
            throw std::invalid_argument("[CKKSEncoder::CKKSEncoder] Encryption parameters are not set correctly.");
        }
        
        ContextDataPointer context_data = context->first_context_data().value();
        const EncryptionParameters& parms = context_data->parms();

        if (parms.scheme() != SchemeType::CKKS) {
            throw std::invalid_argument("[CKKSEncoder::CKKSEncoder] Unsupported scheme.");
        }

        size_t coeff_count = parms.poly_modulus_degree();
        size_t slots = coeff_count / 2;
        Array<size_t> matrix_reps_index_map;

        int logn_int = utils::get_power_of_two(static_cast<uint64_t>(coeff_count));
        if (logn_int < 0) {
            throw std::invalid_argument("[CKKSEncoder::CKKSEncoder] Slots must be a power of two.");
        }
        size_t logn = static_cast<size_t>(logn_int);
        matrix_reps_index_map = Array<size_t>(coeff_count, false, nullptr);
        size_t m = coeff_count << 1;
        size_t gen = utils::GALOIS_GENERATOR; size_t pos = 1;
        for (size_t i = 0; i < slots; i++) {
            size_t index1 = (pos - 1) >> 1;
            size_t index2 = (m - pos - 1) >> 1;
            matrix_reps_index_map[i] = utils::reverse_bits_uint64(static_cast<uint64_t>(index1), logn);
            matrix_reps_index_map[i + slots] = utils::reverse_bits_uint64(static_cast<uint64_t>(index2), logn);
            pos = (pos * gen) & (m - 1);
        }

        // We need 1~(n-1)-th powers of the primitive 2n-th root, m = 2n
        Array<complex<double>> root_powers(coeff_count, false, nullptr);
        Array<complex<double>> inv_root_powers(coeff_count, false, nullptr);

        // Powers of the primitive 2n-th root have 4-fold symmetry
        if (m >= 8) {
            ComplexRoots complex_roots(m);
            for (size_t i = 1; i < coeff_count; i++) {
                root_powers[i] = complex_roots.get_root(
                    static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i), logn)));
                inv_root_powers[i] = conj(complex_roots.get_root(
                    static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i - 1), logn)) + 1));
            }
        } else if (m == 4) {
            root_powers[1] = complex<double>(0, 1);
            inv_root_powers[1] = complex<double>(0, -1);
        } else {
            throw std::invalid_argument("[CKKSEncoder::CKKSEncoder] Poly modulus degree is too small.");
        }

        this->context_ = context;
        this->slots_ = slots;
        this->device = false;
        this->root_powers_ = std::move(root_powers);
        this->inv_root_powers_ = std::move(inv_root_powers);
        this->matrix_reps_index_map = std::move(matrix_reps_index_map);
    }

    __global__ static void kernel_multiply_complex_scalar(Slice<CustomComplex> operand, double fix) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= operand.size()) return;
        operand[index] = operand[index].mul(fix);
    }

    static void multiply_complex_scalar(Slice<complex<double>> operand, double fix) {
        bool device = operand.on_device();
        if (!device) {
            for (size_t i = 0; i < operand.size(); i++) {
                operand[i] *= fix;
            }
        } else {
            size_t total = operand.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(operand.device_index());
            kernel_multiply_complex_scalar<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                CustomComplex::slice(operand), fix
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_multiply_double_scalar(Slice<double> operand, double fix) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= operand.size()) return;
        operand[index] *= fix;
    }

    static void multiply_double_scalar(Slice<double> operand, double fix) {
        bool device = operand.on_device();
        if (!device) {
            for (size_t i = 0; i < operand.size(); i++) {
                operand[i] *= fix;
            }
        } else {
            size_t total = operand.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(operand.device_index());
            kernel_multiply_double_scalar<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                operand, fix
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_fft_transform_from_rev_layer(
        size_t layer, Slice<CustomComplex> operand, size_t logn, ConstSlice<CustomComplex> roots
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t upperbound = 1 << (logn - 1);
        if (i >= upperbound) return;
        size_t m = 1 << (logn - 1 - layer);
        size_t gap = 1 << layer;
        size_t rid = (1 << logn) - (m << 1) + 1 + (i >> layer);
        size_t coeff_index = ((i >> layer) << (layer + 1)) + (i & (gap - 1));
        CustomComplex& x = operand[coeff_index];
        CustomComplex& y = operand[coeff_index + gap];
        CustomComplex u = x;
        CustomComplex v = y;
        const CustomComplex& r = roots[rid];
        x = u.add(v);
        y = u.sub(v).mul(r);
    }

    static void fft_transform_from_rev_layer(size_t layer, Slice<complex<double>> operand, size_t logn, ConstSlice<complex<double>> roots) {
        bool device = operand.on_device();
        if (!device) {
            size_t upperbound = 1 << (logn - 1);
            size_t m = 1 << (logn - 1 - layer);
            size_t gap = 1 << layer;
            for (size_t i = 0; i < upperbound; i++) {
                size_t rid = (1 << logn) - (m << 1) + 1 + (i >> layer);
                size_t coeff_index = ((i >> layer) << (layer + 1)) + (i & (gap - 1));
                complex<double>& x = operand[coeff_index];
                complex<double>& y = operand[coeff_index + gap];
                complex<double> u = x;
                complex<double> v = y;
                const complex<double>& r = roots[rid];
                x = u + v;
                y = (u - v) * r;
            }
        } else {
            size_t total = 1 << (logn - 1);
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(operand.device_index());
            kernel_fft_transform_from_rev_layer<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                layer, CustomComplex::slice(operand), logn, CustomComplex::slice(roots)
            );
            utils::stream_sync();
        }
    }

    static void fft_transform_from_rev(Slice<complex<double>> operand, size_t logn, ConstSlice<complex<double>> roots, double fix) {
        // TODO: Improve this like NTT (with 8 layers into one kernel with __syncthreads())
        bool device = operand.on_device();
        if (device != roots.on_device()) {
            throw std::invalid_argument("[CKKSEncoder::fft_transform_from_rev] operand and roots must be on the same device.");
        }
        size_t n = 1 << logn;
        size_t m = n >> 1; size_t layer = 0;
        for (; m >= 1; m >>= 1) {
            fft_transform_from_rev_layer(layer, operand, logn, roots);
            layer++;
        }
        if (fix != 1.0) {
            multiply_complex_scalar(operand, fix);
        }
    }

    __global__ static void kernel_fft_transform_to_rev_layer(
        size_t layer, Slice<CustomComplex> operand, size_t logn, ConstSlice<CustomComplex> roots
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (1 << (logn - 1))) return;
        size_t m = 1 << layer;
        size_t gap_power = logn - 1 - layer;
        size_t gap = 1 << gap_power;
        size_t rid = m + (i >> gap_power);
        size_t coeff_index = ((i >> gap_power) << (gap_power + 1)) + (i & (gap - 1));
        CustomComplex& x = operand[coeff_index];
        CustomComplex& y = operand[coeff_index + gap];
        const CustomComplex& r = roots[rid];
        CustomComplex u = x;
        CustomComplex v = y.mul(r);
        x = u.add(v);
        y = u.sub(v);
    }

    static void fft_transform_to_rev_layer(size_t layer, Slice<complex<double>> operand, size_t logn, ConstSlice<complex<double>> roots) {
        bool device = operand.on_device();
        if (!device) {
            size_t upperbound = 1 << (logn - 1);
            size_t m = 1 << layer;
            size_t gap_power = logn - 1 - layer;
            size_t gap = 1 << gap_power;
            for (size_t i = 0; i < upperbound; i++) {
                size_t rid = m + (i >> gap_power);
                size_t coeff_index = ((i >> gap_power) << (gap_power + 1)) + (i & (gap - 1));
                complex<double>& x = operand[coeff_index];
                complex<double>& y = operand[coeff_index + gap];
                const complex<double>& r = roots[rid];
                complex<double> u = x;
                complex<double> v = y * r;
                x = u + v;
                y = u - v;
            }
        } else {
            size_t total = 1 << (logn - 1);
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(operand.device_index());
            kernel_fft_transform_to_rev_layer<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                layer, CustomComplex::slice(operand), logn, CustomComplex::slice(roots)
            );
            utils::stream_sync();
        }
    }

    static void fft_transform_to_rev(Slice<complex<double>> operand, size_t logn, ConstSlice<complex<double>> roots) {
        // TODO: Improve this like NTT (with 8 layers into one kernel with __syncthreads())
        bool device = operand.on_device();
        if (device != roots.on_device()) {
            throw std::invalid_argument("[CKKSEncoder::fft_transform_to_rev] operand and roots must be on the same device.");
        }
        size_t m = 1; size_t layer = 0;
        size_t n = 1 << logn;
        for (; m <= (n >> 1); m <<= 1) {
            fft_transform_to_rev_layer(layer, operand, logn, roots);
            layer++;
        }
    }

    __global__ static void kernel_set_conjugate_values(
        ConstSlice<CustomComplex> from, ConstSlice<size_t> index_map, Slice<CustomComplex> target
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= from.size() * 2) return;
        size_t count = target.size() / 2;
        if (index < from.size()) {
            target[index_map[index]] = from[index];
        } else {
            index -= from.size();
            target[index_map[index + count]] = conj(from[index]);
        }
    }

    static void set_conjugate_values(ConstSlice<complex<double>> from, ConstSlice<size_t> index_map, Slice<complex<double>> target) {
        bool device = from.on_device();
        if (!utils::device_compatible(from, target, index_map)) {
            throw std::invalid_argument("[CKKSEncoder::set_conjugate_values] from and target must be on the same device.");
        }
        size_t count = target.size() / 2;
        if (!device) {
            for (size_t i = 0; i < from.size(); i++) {
                target[index_map[i]] = from[i];
                target[index_map[i + count]] = conj(from[i]);
            }
        } else {
            size_t block_size = utils::ceil_div(from.size() * 2, utils::KERNEL_THREAD_COUNT);
            utils::set_device(target.device_index());
            kernel_set_conjugate_values<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                CustomComplex::slice(from),
                index_map,
                CustomComplex::slice(target)
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_retrieve_conjugate_values(
        ConstSlice<CustomComplex> from, ConstSlice<size_t> index_map, Slice<CustomComplex> target
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= target.size()) return;
        target[index] = from[index_map[index]];
    }

    static void retrieve_conjugate_values(ConstSlice<complex<double>> from, ConstSlice<size_t> index_map, Slice<complex<double>> target) {
        bool device = from.on_device();
        if (!utils::device_compatible(from, target, index_map)) {
            throw std::invalid_argument("[CKKSEncoder::retrieve_conjugate_values] from and target must be on the same device.");
        }
        size_t count = target.size();
        if (!device) {
            for (size_t i = 0; i < count; i++) {
                target[i] = from[index_map[i]];
            }
        } else {
            size_t block_size = utils::ceil_div(count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(target.device_index());
            kernel_retrieve_conjugate_values<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                CustomComplex::slice(from),
                index_map,
                CustomComplex::slice(target)
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_gather_real(ConstSlice<CustomComplex> complex_array, Slice<double> real_array) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= complex_array.size()) return;
        real_array[index] = complex_array[index].real;
    }

    static void gather_real(ConstSlice<complex<double>> complex_array, Slice<double> real_array) {
        bool device = complex_array.on_device();
        if (!utils::device_compatible(complex_array, real_array)) {
            throw std::invalid_argument("[CKKSEncoder::gather_real] complex_array and real_array must be on the same device.");
        }
        size_t n = complex_array.size();
        if (n != real_array.size()) {
            throw std::invalid_argument("[CKKSEncoder::gather_real] complex_array and real_array must have the same size.");
        }
        if (!device) {
            for (size_t i = 0; i < n; i++) {
                real_array[i] = complex_array[i].real();
            }
        } else {
            size_t block_size = utils::ceil_div(n, utils::KERNEL_THREAD_COUNT);
            utils::set_device(complex_array.device_index());
            kernel_gather_real<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                CustomComplex::slice(complex_array),
                real_array
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_set_plaintext_value_array_64bits(
        size_t coeff_count, ConstSlice<double> real_values, ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= real_values.size() * coeff_modulus.size()) return;
        size_t i = index % real_values.size();
        size_t j = index / real_values.size();
        double coeffd = real_values[i];
        bool is_negative = coeffd < 0;
        uint64_t coeffu = static_cast<uint64_t>(is_negative ? -coeffd : coeffd);
        if (is_negative) {
            destination[i + j * coeff_count] = utils::negate_uint64_mod(
                coeff_modulus[j].reduce(coeffu), coeff_modulus[j]
            );
        } else {
            destination[i + j * coeff_count] = coeff_modulus[j].reduce(coeffu);
        }
    }

    static void set_plaintext_value_array_64bits(size_t coeff_count, ConstSlice<double> real_values, ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination) {
        bool device = real_values.on_device();
        if (!device) {
            for (size_t i = 0; i < real_values.size(); i++) {
                double coeffd = real_values[i];
                bool is_negative = coeffd < 0;
                uint64_t coeffu = static_cast<uint64_t>(is_negative ? -coeffd : coeffd);
                if (is_negative) {
                    for (size_t j = 0; j < coeff_modulus.size(); j++) {
                        destination[i + j * coeff_count] = utils::negate_uint64_mod(
                            coeff_modulus[j].reduce(coeffu), coeff_modulus[j]
                        );
                    }
                } else {
                    for (size_t j = 0; j < coeff_modulus.size(); j++) {
                        destination[i + j * coeff_count] = coeff_modulus[j].reduce(coeffu);
                    }
                }
            }
        } else {
            size_t total = real_values.size() * coeff_modulus.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(real_values.device_index());
            kernel_set_plaintext_value_array_64bits<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                coeff_count, real_values, coeff_modulus, destination
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_set_plaintext_value_array_128bits(
        size_t coeff_count, ConstSlice<double> real_values, ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= real_values.size() * coeff_modulus.size()) return;
        size_t i = index % real_values.size();
        size_t j = index / real_values.size();
        double coeffd = real_values[i];
        bool is_negative = coeffd < 0;
        __uint128_t coeffu = static_cast<__uint128_t>(is_negative ? -coeffd : coeffd);
        if (is_negative) {
            destination[i + j * coeff_count] = utils::negate_uint64_mod(
                coeff_modulus[j].reduce_uint128(coeffu), coeff_modulus[j]
            );
        } else {
            destination[i + j * coeff_count] = coeff_modulus[j].reduce_uint128(coeffu);
        }
    }
    
    static void set_plaintext_value_array_128bits(size_t coeff_count, ConstSlice<double> real_values, ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination) {
        bool device = real_values.on_device();
        if (!device) {
            for (size_t i = 0; i < real_values.size(); i++) {
                double coeffd = real_values[i];
                bool is_negative = coeffd < 0;
                __uint128_t coeffu = static_cast<__uint128_t>(is_negative ? -coeffd : coeffd);
                if (is_negative) {
                    for (size_t j = 0; j < coeff_modulus.size(); j++) {
                        destination[i + j * coeff_count] = utils::negate_uint64_mod(
                            coeff_modulus[j].reduce_uint128(coeffu), coeff_modulus[j]
                        );
                    }
                } else {
                    for (size_t j = 0; j < coeff_modulus.size(); j++) {
                        destination[i + j * coeff_count] = coeff_modulus[j].reduce_uint128(coeffu);
                    }
                }
            }
        } else {
            size_t total = real_values.size() * coeff_modulus.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(real_values.device_index());
            kernel_set_plaintext_value_array_128bits<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                coeff_count, real_values, coeff_modulus, destination
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_decompose_double_absolute_array(
        ConstSlice<double> real_values,
        size_t n,
        size_t coeff_modulus_size,
        Slice<uint64_t> destination
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n) return;
        double coeffd = abs(round(real_values[index]));
        size_t offset = index * coeff_modulus_size;
        double two_pow_64 = std::pow(2.0, 64.0);
        while (coeffd >= 1) {
            destination[offset] = static_cast<uint64_t>(std::fmod(coeffd, two_pow_64));
            coeffd /= two_pow_64;
            offset++;
        }
    }

    static void decompose_double_absolute_array(
        ConstSlice<double> real_values,
        size_t coeff_modulus_size,
        Slice<uint64_t> destination
    ) {
        bool device = real_values.on_device();
        if (!device) {
            double two_pow_64 = std::pow(2.0, 64.0);
            for (size_t i = 0; i < real_values.size(); i++) {
                double coeffd = std::abs(std::round(real_values[i]));
                size_t offset = i * coeff_modulus_size;
                while (coeffd >= 1) {
                    destination[offset] = static_cast<uint64_t>(std::fmod(coeffd, two_pow_64));
                    coeffd /= two_pow_64;
                    offset++;
                }
            }
        } else {
            size_t total = real_values.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(real_values.device_index());
            kernel_decompose_double_absolute_array<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                real_values, total, coeff_modulus_size, destination
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_set_decomposed_value_array(
        ConstSlice<double> real_values, ConstSlice<uint64_t> decomposed_values,
        size_t n, ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n * coeff_modulus.size()) return;
        size_t i = index % n;
        size_t j = index / n;
        double coeffd = std::round(real_values[i]);
        bool is_negative = coeffd < 0;
        if (is_negative) {
            destination[index] = utils::negate_uint64_mod(
                decomposed_values[index], coeff_modulus[j]
            );
        } else {
            destination[index] = decomposed_values[index];
        }
    }

    static void set_decomposed_value_array(
        ConstSlice<double> real_values, ConstSlice<uint64_t> decomposed_values,
        size_t n, ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination
    ) {
        bool device = real_values.on_device();
        if (!device) {
            for (size_t i = 0; i < n; i++) {
                double coeffd = std::round(real_values[i]);
                bool is_negative = coeffd < 0;
                for (size_t j = 0; j < coeff_modulus.size(); j++) {
                    if (is_negative) {
                        destination[i + j * n] = utils::negate_uint64_mod(
                            decomposed_values[i + j * n], coeff_modulus[j]
                        );
                    } else {
                        destination[i + j * n] = decomposed_values[i + j * n];
                    }
                }
            }
        } else {
            size_t total = n * coeff_modulus.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(real_values.device_index());
            kernel_set_decomposed_value_array<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                real_values, decomposed_values, n, coeff_modulus, destination
            );
            utils::stream_sync();
        }
    }

    static void set_plaintext_value_array_general(
        ContextDataPointer context_data,
        size_t coeff_count, ConstSlice<double> real_values, 
        ConstSlice<Modulus> coeff_modulus, Slice<uint64_t> destination,
        MemoryPoolHandle pool
    ) {
        bool device = real_values.on_device();
        size_t coeff_modulus_size = coeff_modulus.size();
        Array<uint64_t> coeffu_array(coeff_modulus_size * coeff_count, device, pool);
        decompose_double_absolute_array(
            real_values, coeff_modulus_size, coeffu_array.reference()
        );
        context_data->rns_tool().base_q().decompose_array(coeffu_array.reference(), pool);
        set_decomposed_value_array(
            real_values, coeffu_array.const_reference(), coeff_count,
            coeff_modulus, destination
        );
    }
    

    static void set_plaintext_value_array(ContextDataPointer context_data, size_t coeff_count, ConstSlice<double> real_values, ConstSlice<Modulus> coeff_modulus, Plaintext& destination, MemoryPoolHandle pool) {
        size_t coeff_modulus_size = coeff_modulus.size();

        // Verify that the values are not too large to fit in coeff_modulus
        // Note that we have an extra + 1 for the sign bit
        // Don't compute logarithmis of numbers less than 1
        double max_coeff = reduction::max(real_values, pool);
        size_t max_coeff_bit_count = static_cast<size_t>(std::ceil(std::log2(std::max(max_coeff, 1.0))));

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);
        destination.poly().set_zero();

        // Use faster decomposition methods when possible
        if (max_coeff_bit_count <= 64) {
            set_plaintext_value_array_64bits(coeff_count, real_values, coeff_modulus, destination.poly());
        } else if (max_coeff_bit_count <= 128) {
            set_plaintext_value_array_128bits(coeff_count, real_values, coeff_modulus, destination.poly());
        } else {
            set_plaintext_value_array_general(context_data, coeff_count, real_values, coeff_modulus, destination.poly(), pool);
        }
    }
    
    void CKKSEncoder::encode_internal_complex_simd_slice(ConstSlice<std::complex<double>> values, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const {
        
        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_complex_simd_slice] Memory pool is not compatible with device.");
        }

        // if values and this is not on the same device, convert first
        if (this->on_device() && (!values.on_device() || values.device_index() != this->device_index())) {
            Array<std::complex<double>> values_device = Array<std::complex<double>>::create_and_copy_from_slice(values, true, pool);
            encode_internal_complex_simd_slice(values_device.const_reference(), parms_id, scale, destination, pool);
            return;
        } else if (!this->on_device() && values.on_device()) {
            Array<std::complex<double>> values_host = Array<std::complex<double>>::create_and_copy_from_slice(values, false, nullptr);
            encode_internal_complex_simd_slice(values_host.const_reference(), parms_id, scale, destination, pool);
            return;
        }
        
        // check compatible
        if (!utils::device_compatible(values, *this)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Values and destination are not compatible.");
        }
        
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_complex_array] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        size_t slots = this->slot_count();
        if (values.size() > slots) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_complex_array] Too many input values.");
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Check that scale is positive and not too large
        if (scale <= 0 || std::log2(scale) + 1.0 >= static_cast<double>(context_data->total_coeff_modulus_bit_count())) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_complex_array] scale out of bounds.");
        }

        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        if (ntt_tables.size() != coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_complex_array] NTTTables count not correct.");
        }

        bool device = this->on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_complex_array] destination and context_data must be on the same device.");
        }

        size_t n = slots * 2;
        Array<complex<double>> conj_values(n, device, pool);
        set_conjugate_values(
            values,
            this->matrix_reps_index_map.const_reference(),
            conj_values.reference()
        );

        size_t logn = utils::get_power_of_two(n);
        double fix = scale / static_cast<double>(n);
        fft_transform_from_rev(
            conj_values.reference(), logn, 
            this->inv_root_powers_.const_reference(), fix
        );

        Array<double> real_values(n, device, pool);
        gather_real(conj_values.const_reference(), real_values.reference());

        set_plaintext_value_array(context_data, coeff_count, real_values.const_reference(), coeff_modulus, destination, pool);
        
        // Transform to NTT domain
        utils::ntt_inplace_p(destination.poly(), coeff_count, ntt_tables);

        destination.parms_id() = parms_id;
        destination.scale() = scale;
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
        destination.coeff_count() = coeff_count;
    }

    void CKKSEncoder::encode_internal_double_polynomial_slice(utils::ConstSlice<double> values, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const {
        
        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial_slice] Memory pool is not compatible with device.");
        }

        // if values and this is not on the same device, convert first
        if (this->on_device() && (!values.on_device() || values.device_index() != this->device_index())) {
            Array<double> values_device = Array<double>::create_and_copy_from_slice(values, true, pool);
            encode_internal_double_polynomial_slice(values_device.const_reference(), parms_id, scale, destination, pool);
            return;
        } else if (!this->on_device() && values.on_device()) {
            Array<double> values_host = Array<double>::create_and_copy_from_slice(values, false, nullptr);
            encode_internal_double_polynomial_slice(values_host.const_reference(), parms_id, scale, destination, pool);
            return;
        }
        
        // check compatible
        if (!utils::device_compatible(values, *this)) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial_slice] Values and destination are not compatible.");
        }
        
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        size_t slots = this->slot_count();
        if (values.size() > slots * 2) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial] Too many input values.");
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Check that scale is positive and not too large
        if (scale <= 0 || std::log2(scale) + 1.0 >= static_cast<double>(context_data->total_coeff_modulus_bit_count())) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial] scale out of bounds.");
        }

        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        if (ntt_tables.size() != coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial] NTTTables count not correct.");
        }

        bool device = this->on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_polynomial] destination and context_data must be on the same device.");
        }
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);
        destination.poly().set_zero();
        
        size_t n = slots * 2;
        Array<double> real_values(n, device, pool);
        real_values.slice(0, values.size()).copy_from_slice(values);
        multiply_double_scalar(real_values.reference(), scale);

        set_plaintext_value_array(context_data, coeff_count, real_values.const_reference(), coeff_modulus, destination, pool);

        // Transform to NTT domain
        utils::ntt_inplace_p(destination.poly(), coeff_count, ntt_tables);

        destination.parms_id() = parms_id;
        destination.scale() = scale;
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
        destination.coeff_count() = coeff_count;
    }

    __global__ static void kernel_broadcast_double(double d, Slice<double> destination) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= destination.size()) return;
        destination[index] = d;
    }

    static void broadcast_double(double d, Slice<double> destination) {
        bool device = destination.on_device();
        if (!device) {
            for (size_t i = 0; i < destination.size(); i++) {
                destination[i] = d;
            }
        } else {
            size_t total = destination.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_broadcast_double<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                d, destination
            );
            utils::stream_sync();
        }
    }

    /*
    __global__ static void kernel_broadcast_integer(int64_t i, Slice<int64_t> destination) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= destination.size()) return;
        destination[index] = i;
    }

    static void broadcast_integer(int64_t value, Slice<int64_t> destination) {
        bool device = destination.on_device();
        if (!device) {
            for (size_t i = 0; i < destination.size(); i++) {
                destination[i] = value;
            }
        } else {
            size_t total = destination.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            kernel_broadcast_integer<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                value, destination
            );
            utils::stream_sync();
        }
    }
    */

    void CKKSEncoder::encode_internal_double_single(double value, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const {
        
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_single] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        size_t slots = this->slot_count();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Check that scale is positive and not too large
        if (scale <= 0 || std::log2(scale) + 1.0 >= static_cast<double>(context_data->total_coeff_modulus_bit_count())) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_single] scale out of bounds.");
        }

        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        if (ntt_tables.size() != coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_single] NTTTables count not correct.");
        }

        bool device = this->on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_double_single] destination and context_data must be on the same device.");
        }
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);
        destination.poly().set_zero();

        value *= scale;
        
        size_t n = slots * 2;
        Array<double> real_values(n, device, pool);
        broadcast_double(value, real_values.reference());

        set_plaintext_value_array(context_data, coeff_count, real_values.const_reference(), coeff_modulus, destination, pool);

        destination.parms_id() = parms_id;
        destination.scale() = scale;
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
        destination.coeff_count() = coeff_count;
    }

    __global__ static void kernel_reduce_values(
        ConstSlice<int64_t> values, size_t coeff_count, ConstSlice<Modulus> modulus, Slice<uint64_t> destination
    ) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= values.size() * modulus.size()) return;
        size_t i = index % values.size();
        size_t j = index / values.size();
        int64_t value = values[i];
        destination[i + j * coeff_count] = (value < 0) ?
            modulus[j].value() - modulus[j].reduce(static_cast<uint64_t>(-value)) :
            modulus[j].reduce(static_cast<uint64_t>(value));
    }

    static void reduce_values(ConstSlice<int64_t> values, size_t coeff_count, ConstSlice<Modulus> modulus, Slice<uint64_t> destination) {
        bool device = values.on_device();
        if (!device) {
            for (size_t i = 0; i < values.size(); i++) {
                int64_t value = values[i];
                for (size_t j = 0; j < modulus.size(); j++) {
                    destination[i + j * coeff_count] = (value < 0) ?
                        modulus[j].value() - modulus[j].reduce(static_cast<uint64_t>(-value)) :
                        modulus[j].reduce(static_cast<uint64_t>(value));
                }
            }
        } else {
            size_t total = values.size() * modulus.size();
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(values.device_index());
            kernel_reduce_values<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                values, coeff_count, modulus, destination
            );
            utils::stream_sync();
        }

    }
    
    void CKKSEncoder::encode_internal_integer_polynomial_slice(utils::ConstSlice<int64_t> values, ParmsID parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
        
        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Memory pool is not compatible with device.");
        }

        // if values and this is not on the same device, convert first
        if (this->on_device() && (!values.on_device() || values.device_index() != this->device_index())) {
            Array<int64_t> values_device = Array<int64_t>::create_and_copy_from_slice(values, true, pool);
            encode_internal_integer_polynomial_slice(values_device.const_reference(), parms_id, destination, pool);
            return;
        } else if (!this->on_device() && values.on_device()) {
            Array<int64_t> values_host = Array<int64_t>::create_and_copy_from_slice(values, false, nullptr);
            encode_internal_integer_polynomial_slice(values_host.const_reference(), parms_id, destination, pool);
            return;
        }
        
        // check compatible
        if (!utils::device_compatible(values, *this)) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_polynomial_slice] Values and destination are not compatible.");
        }
        
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_polynomial_slice] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        if (ntt_tables.size() != coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_polynomial] NTTTables count not correct.");
        }

        bool device = this->on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_polynomial] destination and context_data must be on the same device.");
        }
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);
        destination.poly().set_zero();

        reduce_values(
            values,
            coeff_count, coeff_modulus, destination.poly()
        );

        // Transform to NTT domain
        utils::ntt_inplace_p(destination.poly(), coeff_count, ntt_tables);

        destination.parms_id() = parms_id;
        destination.scale() = 1.0;
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
        destination.coeff_count() = coeff_count;
    }

    void CKKSEncoder::encode_internal_integer_single(int64_t value, ParmsID parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_single] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        size_t slots = this->slot_count();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        if (ntt_tables.size() != coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_single] NTTTables count not correct.");
        }

        bool device = this->on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[CKKSEncoder::encode_internal_integer_single] destination and context_data must be on the same device.");
        }
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);
        destination.poly().set_zero();
        
        size_t n = slots * 2;
        Array<int64_t> values(n, false, nullptr); values[0] = value;
        if (device) values.to_device_inplace(pool);
        reduce_values(
            values.const_reference(),
            coeff_count, coeff_modulus, destination.poly()
        );

        // Transform to NTT domain
        utils::ntt_inplace_p(destination.poly(), coeff_count, ntt_tables);

        destination.parms_id() = parms_id;
        destination.scale() = 1.0;
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
        destination.coeff_count() = coeff_count;
    }

    __global__ static void kernel_accumulate_complex(
        ConstSlice<uint64_t> inputs, size_t coeff_count, ConstSlice<uint64_t> decryption_modulus,
        size_t coeff_modulus_size,
        ConstSlice<uint64_t> upper_half_threshold, Slice<CustomComplex> destination,
        double inv_scale
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        double two_pow_64 = std::pow(2.0, 64.0);
        ConstSlice<uint64_t> input = inputs.const_slice(i * coeff_modulus_size, (i + 1) * coeff_modulus_size);
        if (utils::is_greater_or_equal_uint(input, upper_half_threshold)) {
            double scaled_two_pow_64 = inv_scale;
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                if (input[j] > decryption_modulus[j]) {
                    uint64_t diff = input[j] - decryption_modulus[j];
                    if (diff != 0) destination[i].real += diff * scaled_two_pow_64;
                } else {
                    uint64_t diff = decryption_modulus[j] - input[j];
                    if (diff != 0) destination[i].real -= diff * scaled_two_pow_64;
                }
                scaled_two_pow_64 *= two_pow_64;
            }
        } else {
            double scaled_two_pow_64 = inv_scale;
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                uint64_t curr_coeff = input[j];
                if (curr_coeff != 0) destination[i].real += curr_coeff * scaled_two_pow_64;
                scaled_two_pow_64 *= two_pow_64;
            }
        }
    }

    static void accumulate_complex(
        ConstSlice<uint64_t> inputs, size_t coeff_count, ConstSlice<uint64_t> decryption_modulus,
        size_t coeff_modulus_size,
        ConstSlice<uint64_t> upper_half_threshold, Slice<complex<double>> destination,
        double inv_scale
    ) {
        bool device = inputs.on_device();
        if (!device) {
            double two_pow_64 = std::pow(2.0, 64.0);    
            for (size_t i = 0; i < coeff_count; i++) {
                ConstSlice<uint64_t> input = inputs.const_slice(i * coeff_modulus_size, (i + 1) * coeff_modulus_size);
                if (utils::is_greater_or_equal_uint(input, upper_half_threshold)) {
                    double scaled_two_pow_64 = inv_scale;
                    for (size_t j = 0; j < coeff_modulus_size; j++) {
                        if (input[j] > decryption_modulus[j]) {
                            uint64_t diff = input[j] - decryption_modulus[j];
                            if (diff != 0) destination[i] += diff * scaled_two_pow_64;
                        } else {
                            uint64_t diff = decryption_modulus[j] - input[j];
                            if (diff != 0) destination[i] -= diff * scaled_two_pow_64;
                        }
                        scaled_two_pow_64 *= two_pow_64;
                    }
                } else {
                    double scaled_two_pow_64 = inv_scale;
                    for (size_t j = 0; j < coeff_modulus_size; j++) {
                        uint64_t curr_coeff = input[j];
                        if (curr_coeff != 0) destination[i] += curr_coeff * scaled_two_pow_64;
                        scaled_two_pow_64 *= two_pow_64;
                    }
                }
            }
        } else {
            size_t total = coeff_count;
            size_t block_size = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(inputs.device_index());
            kernel_accumulate_complex<<<block_size, utils::KERNEL_THREAD_COUNT>>>(
                inputs, coeff_count, decryption_modulus, coeff_modulus_size, upper_half_threshold, 
                CustomComplex::slice(destination), inv_scale
            );
            utils::stream_sync();
        }
    }
    
    void CKKSEncoder::decode_internal_simd_slice(const Plaintext& plain, Slice<std::complex<double>> destination, MemoryPoolHandle pool) const {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_simd_slice] Plaintext is not in NTT form.");
        }
        size_t slots = this->slot_count();
        bool device = plain.on_device();
        if (destination.size() != slots) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_simd_slice] Destination size is not correct.");
        }

        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(plain.parms_id());
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_simd_slice] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        
        if (!utils::same(device, this->on_device(), context_data->on_device(), plain.on_device())) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_simd_slice] Operands must be on the same device.");
        }

        ConstSlice<uint64_t> decryption_modulus = context_data->total_coeff_modulus();
        ConstSlice<uint64_t> upper_half_threshold = context_data->upper_half_threshold();
        size_t logn = utils::get_power_of_two(coeff_count);

        double inv_scale = 1.0 / plain.scale();
        if (plain.data().size() != coeff_count * coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_simd_slice] Plaintext data length is not correct.");
        }
        Array<uint64_t> plain_copy = plain.data().get_inner().clone(pool);

        // Transform each polynomial from NTT domain
        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::intt_inplace_p(plain_copy.reference(), coeff_count, ntt_tables);

        // CRT-compose the polynomial
        context_data->rns_tool().base_q().compose_array(plain_copy.reference(), pool);

        // Create floating-point representations of the multi-precision integer coefficients
        Array<complex<double>> res(coeff_count, device, pool);
        accumulate_complex(
            plain_copy.const_reference(), coeff_count, decryption_modulus,
            coeff_modulus_size, upper_half_threshold, res.reference(), inv_scale
        );

        fft_transform_to_rev(
            res.reference(), logn, 
            this->root_powers_.const_reference()
        );

        if (!device) {
            retrieve_conjugate_values(
                res.const_reference(),
                this->matrix_reps_index_map.const_reference(),
                destination
            );
        } else {
            Array<complex<double>> destination_device(destination.size(), true, pool);
            retrieve_conjugate_values(
                res.const_reference(),
                this->matrix_reps_index_map.const_reference(),
                destination_device.reference()
            );
            destination_device.to_host_inplace();
            destination.copy_from_slice(
                destination_device.const_reference()
            );
        }
    }

    void CKKSEncoder::decode_internal_polynomial_slice(const Plaintext& plain, Slice<double> destination, MemoryPoolHandle pool) const {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_polynomial_slice] Plaintext is not in NTT form.");
        }
        size_t slots = this->slot_count();
        if (destination.size() != slots * 2) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_polynomial_slice] Destination size is not correct.");
        }

        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(plain.parms_id());
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_polynomial_slice] parms_id not valid for context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        
        bool device = plain.on_device();
        if (!utils::same(device, this->on_device(), context_data->on_device())) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_polynomial_slice] Operands must be on the same device.");
        }

        ConstSlice<uint64_t> decryption_modulus = context_data->total_coeff_modulus();
        ConstSlice<uint64_t> upper_half_threshold = context_data->upper_half_threshold();

        double inv_scale = 1.0 / plain.scale();
        if (plain.data().size() != coeff_count * coeff_modulus_size) {
            throw std::invalid_argument("[CKKSEncoder::decode_internal_polynomial_slice] Plaintext data length is not correct.");
        }
        Array<uint64_t> plain_copy = plain.data().get_inner().clone(pool);

        // Transform each polynomial from NTT domain
        ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::intt_inplace_p(plain_copy.reference(), coeff_count, ntt_tables);

        // CRT-compose the polynomial
        context_data->rns_tool().base_q().compose_array(plain_copy.reference(), pool);

        // Create floating-point representations of the multi-precision integer coefficients
        Array<complex<double>> res(coeff_count, device, pool);
        accumulate_complex(
            plain_copy.const_reference(), coeff_count, decryption_modulus,
            coeff_modulus_size, upper_half_threshold, res.reference(), inv_scale
        );

        Array<double> real_values(coeff_count, device, pool);
        gather_real(res.const_reference(), real_values.reference());

        destination.copy_from_slice(real_values.const_reference());
    }

}