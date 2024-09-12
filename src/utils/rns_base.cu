#include "constants.h"
#include "poly_small_mod.h"
#include "rns_base.h"
#include "polynomial_buffer.h"

namespace troy {namespace utils {

    void multiply_many_uint64_except(ConstSlice<uint64_t> operands, size_t except, Slice<uint64_t> result) {
        if (operands.on_device() || result.on_device()) {
            throw std::invalid_argument("[multiply_many_uint64_except] Operands and result must be on host memory.");
        }
        size_t count = operands.size();
        set_zero_uint(result);
        if (count == 1 && except == 0) {
            result[0] = 1;
            return;
        }
        result[0] = (except == 0) ? 1 : operands[0];
        Array<uint64_t> temp_mpi(count, false, nullptr);
        for (size_t i = 1; i < count; i++) {
            if (i == except) {
                continue;
            }
            multiply_uint_uint64(result.as_const(), operands[i], temp_mpi.slice(0, i+1));
            set_uint(temp_mpi.const_reference(), i+1, result);
        }
    }

    RNSBase::RNSBase(ConstSlice<Modulus> rnsbase) {

        if (rnsbase.on_device()) {
            throw std::invalid_argument("[RNSBase::RNSBase] RNSBase cannot be created from device memory.");
        }

        if (rnsbase.size() == 0) {
            throw std::invalid_argument("[RNSBase::RNSBase] RNSBase cannot be empty.");
        }

        size_t n = rnsbase.size();
        for (size_t i = 0; i < n; i++) {
            if (rnsbase[i].is_zero()) {
                throw std::invalid_argument("[RNSBase::RNSBase] RNSBase cannot contain zero modulus.");
            }
            for (size_t j = i + 1; j < n; j++) {
                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {
                    throw std::invalid_argument("[RNSBase::RNSBase] RNSBase moduli must be pairwise coprime.");
                }
            }
        }

        this->base_ = Array<Modulus>::create_and_copy_from_slice(rnsbase, nullptr);
        this->initialize();

    }

    void RNSBase::initialize() {

        size_t n = this->base_.size();
        
        Array<uint64_t> base_product(n, false, nullptr);
        Array<uint64_t> punctured_product(n * n, false, nullptr);
        Array<MultiplyUint64Operand> inv_punctured_product_mod_base(n, false, nullptr);
        
        if (n == 1) {
            base_product[0] = this->base_[0].value();
            punctured_product[0] = 1;
            inv_punctured_product_mod_base[0] = MultiplyUint64Operand(1, this->base_[0]);
        } else {
            Array<uint64_t> base_values(n, false, nullptr);
            for (size_t i = 0; i < n; i++) {
                base_values[i] = this->base_[i].value();
            }
            for (size_t i = 0; i < n; i++) {
                multiply_many_uint64_except(
                    base_values.const_reference(),
                    i,
                    punctured_product.slice(i * n, (i + 1) * n)
                );
            }
            multiply_uint_uint64(
                punctured_product.const_slice(0, n),
                this->base_[0].value(),
                base_product.reference()
            );
            bool invertible = true;
            for (size_t i = 0; i < n; i++) {
                uint64_t temp = utils::modulo_uint(
                    punctured_product.const_slice(i * n, (i + 1) * n),
                    this->base_[i]
                );
                uint64_t inv_temp = 0;
                invertible = invertible &&
                    try_invert_uint64_mod(temp, this->base_[i], inv_temp);
                if (!invertible) {
                    throw std::invalid_argument("[RNSBase::initialize] RNSBase product is not invertible.");
                }
                inv_punctured_product_mod_base[i] = MultiplyUint64Operand(inv_temp, this->base_[i]);
            }
        }

        this->base_product_ = std::move(base_product);
        this->punctured_product_ = std::move(punctured_product);
        this->inv_punctured_product_mod_base_ = std::move(inv_punctured_product_mod_base);
        this->device = false;
    }

    void RNSBase::decompose_single(Slice<uint64_t> value) const {
        if (value.size() != base_.size()) {
            throw std::runtime_error("[RNSBase::decompose_single] Invalid size of value.");
        }
        if (value.on_device() || this->on_device()) {
            throw std::invalid_argument("[RNSBase::decompose_single] Value and RNSBase must be on host memory. If you wish to conduct on device, use decompose_array instead.");
        }
        if (this->size() > 1) {
            Array<uint64_t> copied = Array<uint64_t>::create_and_copy_from_slice(value.as_const(), nullptr);
            for (size_t i = 0; i < base_.size(); ++i) {
                value[i] = utils::modulo_uint(copied.const_reference(), this->base_[i]);
            }
        }
    }

    void host_rnsbase_decompose_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        size_t n = self.size();
        size_t count = from.size() / n;
        ConstSlice<Modulus> base = self.base();
        for (size_t i = 0; i < count; i++) {
            ConstSlice<uint64_t> single = from.const_slice(i * n, (i + 1) * n);
            for (size_t j = 0; j < n; j++) {
                result[j * count + i] = utils::modulo_uint(single, base[j]);
            }
        }
    } 

    __device__ void device_rnsbase_decompose_array(ConstSlice<Modulus> base, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= result.size()) {
            return;
        }
        size_t n = base.size();
        size_t count = from.size() / n;
        size_t single_index = global_index / n;
        size_t base_index = global_index % n;
        result[base_index * count + single_index] = utils::modulo_uint(
            from.const_slice(single_index * n, (single_index + 1) * n),
            base[base_index]
        );
    }
    __global__ void kernel_rnsbase_decompose_array(ConstSlice<Modulus> base, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        device_rnsbase_decompose_array(base, from, result);
    }
    __global__ void kernel_rnsbase_decompose_array_batched(ConstSlice<Modulus> base, ConstSliceArrayRef<uint64_t> from, SliceArrayRef<uint64_t> result) {
        for (size_t i = 0; i < from.size(); i++) {
            device_rnsbase_decompose_array(base, from[i], result[i]);
        }
    }

    void rnsbase_decompose_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        if (!utils::device_compatible(self, from, result)) {
            throw std::invalid_argument("[rnsbase_decompose_array] RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(from.size(), KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_rnsbase_decompose_array<<<block_count, KERNEL_THREAD_COUNT>>>(self.base(), from, result);
            utils::stream_sync();
        } else {
            host_rnsbase_decompose_array(self, from, result);
        }
    } 
    
    void rnsbase_decompose_array_batched(const RNSBase& self, const ConstSliceVec<uint64_t>& from, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        if (from.size() != result.size()) {
            throw std::invalid_argument("[rnsbase_decompose_array_batched] Input and output sizes must be the same.");
        }
        if (from.empty()) return;
        bool device = self.on_device();
        if (!device || from.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < from.size(); i++) {
                rnsbase_decompose_array(self, from[i], result[i]);
            }
        } else {
            size_t block_count = ceil_div<size_t>(from[0].size(), KERNEL_THREAD_COUNT);
            auto from_batched = utils::construct_batch(from, pool, self);
            auto result_batched = utils::construct_batch(result, pool, self);
            utils::set_device(self.device_index());
            kernel_rnsbase_decompose_array_batched<<<block_count, KERNEL_THREAD_COUNT>>>(self.base(), from_batched, result_batched);
            utils::stream_sync();
        }
    } 

    void RNSBase::decompose_array(Slice<uint64_t> value, MemoryPoolHandle pool) const {
        if (value.size() % this->size() != 0) {
            throw std::invalid_argument("[RNSBase::decompose_array] Value size must be a multiple of RNSBase size.");
        }
        if (this->size() == 1) {
            return; // nothing to do;
        }
        Array<uint64_t> cloned = Array<uint64_t>::create_and_copy_from_slice(value.as_const(), pool);
        rnsbase_decompose_array(*this, cloned.const_reference(), value);
    }

    void RNSBase::decompose_array_batched(const SliceVec<uint64_t>& value, MemoryPoolHandle pool) const {
        for (size_t i = 0; i < value.size(); i++) {
            if (value[i].size() % this->size() != 0) {
                throw std::invalid_argument("[RNSBase::decompose_array] Value size must be a multiple of RNSBase size.");
            }
        }
        if (this->size() == 1) {
            return; // nothing to do;
        }
        std::vector<Array<uint64_t>> cloned(value.size());
        for (size_t i = 0; i < value.size(); i++) {
            cloned[i] = Array<uint64_t>::create_uninitialized(value[i].size(), value[i].on_device(), pool);
        }
        auto cloned_batched = utils::rcollect_const_reference(cloned);
        utils::copy_slice_b(utils::rcollect_as_const(value), utils::rcollect_reference(cloned));
        rnsbase_decompose_array_batched(*this, cloned_batched, value, pool);
    }

    void RNSBase::compose_single(Slice<uint64_t> value) const {
        if (value.size() != base_.size()) {
            throw std::runtime_error("[RNSBase::compose_single] Invalid size of value.");
        }
        if (value.on_device() || this->on_device()) {
            throw std::invalid_argument("[RNSBase::compose_single] Value and RNSBase must be on host memory. If you wish to conduct on device, use compose_array instead.");
        }
        if (this->size() > 1) {
            Array<uint64_t> copied = Array<uint64_t>::create_and_copy_from_slice(value.as_const(), nullptr);
            set_zero_uint(value);
            Array<uint64_t> temp_mpi(this->size(), false, nullptr);
            for (size_t i = 0; i < base_.size(); ++i) {
                uint64_t temp_prod = utils::multiply_uint64operand_mod(copied[i], this->inv_punctured_product_mod_base()[i], this->base()[i]);
                utils::multiply_uint_uint64(this->punctured_product(i), temp_prod, temp_mpi.reference());
                utils::add_uint_mod_inplace(value, temp_mpi.const_reference(), this->base_product());
            }
        }
    }

    void host_rnsbase_compose_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result, Slice<uint64_t> temp_mpi) {
        size_t n = self.size();
        size_t count = from.size() / n;
        ConstSlice<Modulus> base = self.base();
        for (size_t i = 0; i < count; i++) {
            ConstSlice<uint64_t> single = from.const_slice(i * n, (i + 1) * n);
            Slice<uint64_t> temp_mpi_single = temp_mpi.slice(0, n);
            Slice<uint64_t> result_single = result.slice(i * n, (i + 1) * n);
            set_zero_uint(result_single);
            for (size_t j = 0; j < n; j++) {
                uint64_t temp_prod = utils::multiply_uint64operand_mod(single[j], self.inv_punctured_product_mod_base()[j], base[j]);
                utils::multiply_uint_uint64(self.punctured_product(j), temp_prod, temp_mpi_single);
                utils::add_uint_mod_inplace(result_single, temp_mpi_single.as_const(), self.base_product());
            }
        }
    }

    __global__ void kernel_rnsbase_compose_array(
        ConstSlice<Modulus> self_base,
        ConstSlice<uint64_t> self_base_product,
        ConstSlice<uint64_t> self_punctured_product,
        ConstSlice<MultiplyUint64Operand> self_inv_punctured_product_mod_base,
        ConstSlice<uint64_t> from, Slice<uint64_t> result, Slice<uint64_t> temp_mpi
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t n = self_base.size();
        size_t count = from.size() / n;
        if (global_index >= count) {
            return;
        }
        size_t i = global_index;
        ConstSlice<uint64_t> single = from.const_slice(i * n, (i + 1) * n);
        Slice<uint64_t> temp_mpi_single = temp_mpi.slice(i * n, (i + 1) * n);
        Slice<uint64_t> result_single = result.slice(i * n, (i + 1) * n);
        set_zero_uint(result_single);
        for (size_t j = 0; j < n; j++) {
            uint64_t temp_prod = utils::multiply_uint64operand_mod(single[j], self_inv_punctured_product_mod_base[j], self_base[j]);
            utils::multiply_uint_uint64(self_punctured_product.const_slice(j * n, (j + 1) * n), temp_prod, temp_mpi_single);
            utils::add_uint_mod_inplace(result_single, temp_mpi_single.as_const(), self_base_product);
        }
    }

    void rnsbase_compose_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result, Slice<uint64_t> temp_mpi) {
        if (!utils::device_compatible(self, from, result, temp_mpi)) {
            throw std::invalid_argument("[rnsbase_compose_array] RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(from.size() / self.size(), KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_rnsbase_compose_array<<<block_count, KERNEL_THREAD_COUNT>>>(
                self.base(), self.base_product(), self.punctured_product(), self.inv_punctured_product_mod_base(),
                from, result, temp_mpi
            );
            utils::stream_sync();
        } else {
            host_rnsbase_compose_array(self, from, result, temp_mpi);
        }
    }

    void host_rnsbase_compose_rearrange_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        size_t n = self.size();
        size_t count = from.size() / n;
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < n; j++) {
                result[i * n + j] = from[j * count + i];
            }
        }
    }

    __global__ void kernel_rnsbase_compose_rearrange_array(size_t base_size, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t n = base_size;
        size_t count = from.size() / n;
        if (global_index >= count * n) {
            return;
        }
        size_t i = global_index / n;
        size_t j = global_index % n;
        result[i * n + j] = from[j * count + i];
    }

    void rnsbase_compose_rearrange_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        if (!utils::device_compatible(self, from, result)) {
            throw std::invalid_argument("[rnsbase_compose_rearrange_array] RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(from.size(), KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_rnsbase_compose_rearrange_array<<<block_count, KERNEL_THREAD_COUNT>>>(self.size(), from, result);
            utils::stream_sync();
        } else {
            host_rnsbase_compose_rearrange_array(self, from, result);
        }
    }

    void RNSBase::compose_array(Slice<uint64_t> value, MemoryPoolHandle pool) const {
        if (value.size() % this->size() != 0) {
            throw std::invalid_argument("[RNSBase::compose_array] Value size must be a multiple of RNSBase size.");
        }
        if (this->size() == 1) {
            return; // nothing to do;
        }
        Array<uint64_t> cloned(value.size(), value.on_device(), pool);
        rnsbase_compose_rearrange_array(*this, value.as_const(), cloned.reference());
        Array<uint64_t> temp_mpi(value.size(), value.on_device(), pool);
        rnsbase_compose_array(*this, cloned.const_reference(), value, temp_mpi.reference());
    }

    void host_fast_convert_array(
        const RNSBase& ibase, const RNSBase& obase, 
        ConstSlice<uint64_t> base_change_matrix,
        ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<uint64_t> output
    ) {
        size_t ibase_size = ibase.size();
        size_t count = input.size() / ibase_size;
        for (size_t i = 0; i < ibase.size(); i++) {
            const MultiplyUint64Operand& op = ibase.inv_punctured_product_mod_base()[i];
            const Modulus& base = ibase.base()[i];
            if (op.operand == 1) {
                for (size_t j = 0; j < count; j++) {
                    temp[j * ibase_size + i] = utils::barrett_reduce_uint64(input[i * count + j], base);
                }
            } else {
                for (size_t j = 0; j < count; j++) {
                    temp[j * ibase_size + i] = utils::multiply_uint64operand_mod(input[i * count + j], op, base);
                }
            }
        }
        size_t obase_size = obase.size();
        for (size_t i = 0; i < obase_size; i++) {
            for (size_t j = 0; j < count; j++) {
                output[i * count + j] = utils::dot_product_mod(
                    temp.const_slice(j * ibase_size, (j + 1) * ibase_size),
                    base_change_matrix.const_slice(i * ibase_size, (i + 1) * ibase_size),
                    obase.base()[i]
                );
            }
        }
    }

    __global__ void kernel_fast_convert_array(
        ConstSlice<Modulus> ibase, 
        ConstSlice<Modulus> obase,
        ConstSlice<MultiplyUint64Operand> ibase_inv_punctured_product_mod_base,
        ConstSlice<uint64_t> base_change_matrix,
        ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<uint64_t> output
    ) {
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        size_t ibase_size = ibase.size();
        size_t obase_size = obase.size();
        size_t count = input.size() / ibase_size;
        if (j >= count) {
            return;
        }
        for (size_t i = 0; i < ibase_size; i++) {
            const MultiplyUint64Operand& op = ibase_inv_punctured_product_mod_base[i];
            const Modulus& base = ibase[i];
            if (op.operand == 1) {
                temp[j * ibase_size + i] = utils::barrett_reduce_uint64(input[i * count + j], base);
            } else {
                temp[j * ibase_size + i] = utils::multiply_uint64operand_mod(input[i * count + j], op, base);
            }
        }
        for (size_t i = 0; i < obase_size; i++) {
            output[i * count + j] = utils::dot_product_mod(
                temp.const_slice(j * ibase_size, (j + 1) * ibase_size),
                base_change_matrix.const_slice(i * ibase_size, (i + 1) * ibase_size),
                obase[i]
            );
        }
    }

    void BaseConverter::fast_convert_array(ConstSlice<uint64_t> input, Slice<uint64_t> output, MemoryPoolHandle pool) const {
        size_t ibase_size = this->input_base().size();
        size_t obase_size = this->output_base().size();
        size_t count = input.size() / ibase_size;
        if (input.size() != count * ibase_size) {
            throw std::invalid_argument("[BaseConverter::fast_convert_array] Input size must be a multiple of input base size.");
        }
        if (output.size() != count * obase_size) {
            throw std::invalid_argument("[BaseConverter::fast_convert_array] Output size must be a multiple of output base size.");
        }
        Buffer<uint64_t> temp(ibase_size, count, input.on_device(), pool);
        
        if (!utils::device_compatible(*this, input, temp)) {
            throw std::invalid_argument("[fast_convert_array_step1] RNSBase and value must be on the same memory.");
        }
        bool device = this->on_device();
        if (device) {
            size_t count = input.size() / this->input_base().size();
            size_t block_count = ceil_div<size_t>(count, KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_fast_convert_array<<<block_count, KERNEL_THREAD_COUNT>>>(
                this->input_base().base(), this->output_base().base(), this->input_base().inv_punctured_product_mod_base(),
                this->base_change_matrix(),
                input, temp.reference(), output
            );
            utils::stream_sync();
        } else {
            host_fast_convert_array(this->input_base(), this->output_base(), this->base_change_matrix(), input, temp.reference(), output);
        }
    }

    void host_exact_convey_array_step1(const RNSBase& ibase, ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<double> v) {
        size_t count = input.size() / ibase.size();
        for (size_t i = 0; i < ibase.size(); i++) {
            const MultiplyUint64Operand& op = ibase.inv_punctured_product_mod_base()[i];
            const Modulus& base = ibase.base()[i];
            double divisor = static_cast<double>(base.value());
            if (op.operand == 1) {
                for (size_t j = 0; j < count; j++) {
                    temp[j * ibase.size() + i] = utils::barrett_reduce_uint64(input[i * count + j], base);
                    double dividend = static_cast<double>(temp[j * ibase.size() + i]);
                    v[j * ibase.size() + i] = dividend / divisor;
                }
            } else {
                for (size_t j = 0; j < count; j++) {
                    temp[j * ibase.size() + i] = utils::multiply_uint64operand_mod(input[i * count + j], op, base);
                    double dividend = static_cast<double>(temp[j * ibase.size() + i]);
                    v[j * ibase.size() + i] = dividend / divisor;
                }
            }
        }
    }

    __global__ void kernel_exact_convey_array_step1(
        ConstSlice<Modulus> ibase, 
        ConstSlice<MultiplyUint64Operand> ibase_inv_punctured_product_mod_base,
        ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<double> v
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= input.size()) {
            return;
        }
        size_t n = ibase.size();
        size_t ibase_size = ibase.size();
        size_t count = input.size() / ibase_size;
        size_t i = global_index / count;
        size_t j = global_index % count;
        const MultiplyUint64Operand& op = ibase_inv_punctured_product_mod_base[i];
        const Modulus& base = ibase[i];
        if (op.operand == 1) {
            temp[j * ibase_size + i] = utils::barrett_reduce_uint64(input[global_index], base);
        } else {
            temp[j * ibase_size + i] = utils::multiply_uint64operand_mod(input[global_index], op, base);
        }
        double dividend = static_cast<double>(temp[j * ibase_size + i]);
        v[j * ibase_size + i] = dividend / static_cast<double>(base.value());
    }

    void exact_convey_array_step1(const BaseConverter& self, ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<double> v) {
        if (!utils::device_compatible(self, input, temp, v)) {
            throw std::invalid_argument("[exact_convey_array_step1] RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(input.size(), KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_exact_convey_array_step1<<<block_count, KERNEL_THREAD_COUNT>>>(
                self.input_base().base(), self.input_base().inv_punctured_product_mod_base(),
                input, temp, v
            );
            utils::stream_sync();
        } else {
            host_exact_convey_array_step1(self.input_base(), input, temp, v);
        }
    }

    void host_exact_convey_array_step2(const BaseConverter& self, ConstSlice<uint64_t> temp, ConstSlice<double> v, Slice<uint64_t> output) {
        size_t ibase_size = self.input_base().size();
        size_t count = temp.size() / ibase_size;
        const Modulus& p = self.output_base().base()[0];
        size_t q_mod_p = utils::modulo_uint(self.input_base().base_product(), p);
        for (size_t j = 0; j < count; j++) {
            double aggregated_v = 0;
            for (size_t i = j*ibase_size; i < (j+1)*ibase_size; i++) {
                aggregated_v += v[i];
            }
            uint64_t aggregated_rounded_v = std::round(aggregated_v);
            uint64_t sum_mod_obase = utils::dot_product_mod(
                temp.const_slice(j * ibase_size, (j + 1) * ibase_size),
                self.base_change_matrix(), // because output has only one modulus, the row is the matrix itself.
                p
            );
            uint64_t v_q_mod_p = utils::multiply_uint64_mod(aggregated_rounded_v, q_mod_p, p);
            output[j] = utils::sub_uint64_mod(sum_mod_obase, v_q_mod_p, p);
        }
    }

    __global__ void kernel_exact_convey_array_step2(
        size_t ibase_size,
        ConstSlice<uint64_t> input_base_product,
        ConstSlice<Modulus> obase,
        ConstSlice<uint64_t> base_change_matrix,
        ConstSlice<uint64_t> temp, ConstSlice<double> v, Slice<uint64_t> output
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= output.size()) {
            return;
        }
        size_t obase_size = obase.size();
        const Modulus& p = obase[0];
        size_t count = temp.size() / ibase_size;
        size_t q_mod_p = utils::modulo_uint(input_base_product, p);
        size_t j = global_index;
        double aggregated_v = 0;
        for (size_t i = j*ibase_size; i < (j+1)*ibase_size; i++) {
            aggregated_v += v[i];
        }
        uint64_t aggregated_rounded_v = std::round(aggregated_v);
        uint64_t sum_mod_obase = utils::dot_product_mod(
            temp.const_slice(j * ibase_size, (j + 1) * ibase_size),
            base_change_matrix, // because output has only one modulus, the row is the matrix itself.
            p
        );
        uint64_t v_q_mod_p = utils::multiply_uint64_mod(aggregated_rounded_v, q_mod_p, p);
        output[j] = utils::sub_uint64_mod(sum_mod_obase, v_q_mod_p, p);
    }

    void exact_convey_array_step2(const BaseConverter& self, ConstSlice<uint64_t> temp, ConstSlice<double> v, Slice<uint64_t> output) {
        if (!utils::device_compatible(self, temp, v, output)) {
            throw std::invalid_argument("[exact_convey_array_step2] RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(output.size(), KERNEL_THREAD_COUNT);
            utils::set_device(output.device_index());
            kernel_exact_convey_array_step2<<<block_count, KERNEL_THREAD_COUNT>>>(
                self.input_base().size(),
                self.input_base().base_product(),
                self.output_base().base(),
                self.base_change_matrix(),
                temp, v, output
            );
            utils::stream_sync();
        } else {
            host_exact_convey_array_step2(self, temp, v, output);
        }
    }

    void BaseConverter::exact_convey_array(ConstSlice<uint64_t> input, Slice<uint64_t> output, MemoryPoolHandle pool) const {
        size_t ibase_size = this->input_base().size();
        if (this->output_base().size() != 1) {
            throw std::invalid_argument("[BaseConverter::exact_convey_array] Output base size must be 1.");
        }
        size_t count = input.size() / ibase_size;
        if (input.size() != count * ibase_size) {
            throw std::invalid_argument("[BaseConverter::exact_convey_array] Input size must be a multiple of input base size.");
        }
        if (output.size() < count) {
            throw std::invalid_argument("[BaseConverter::exact_convey_array] Output size is too small");
        }
        Array<uint64_t> temp(count * ibase_size, input.on_device(), pool);
        Array<double> v(count * ibase_size, input.on_device(), pool);
        exact_convey_array_step1(*this, input, temp.reference(), v.reference());
        exact_convey_array_step2(*this, temp.const_reference(), v.const_reference(), output);
    }

}}