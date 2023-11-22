#include "rns.cuh"

namespace troy {namespace utils {

    void multiply_many_uint64_except(ConstSlice<uint64_t> operands, size_t except, Slice<uint64_t> result) {
        size_t count = operands.size();
        set_zero_uint(result);
        if (count == 1 && except == 0) {
            result[0] = 1;
            return;
        }
        result[0] = (except == 0) ? 1 : operands[0];
        Array<uint64_t> temp_mpi(count, false);
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
            throw std::invalid_argument("RNSBase cannot be created from device memory.");
        }

        if (rnsbase.size() == 0) {
            throw std::invalid_argument("RNSBase cannot be empty.");
        }

        size_t n = rnsbase.size();
        for (size_t i = 0; i < n; i++) {
            if (rnsbase[i].is_zero()) {
                throw std::invalid_argument("RNSBase cannot contain zero modulus.");
            }
            for (size_t j = i + 1; j < n; j++) {
                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {
                    throw std::invalid_argument("RNSBase moduli must be pairwise coprime.");
                }
            }
        }

        this->base_ = Array<Modulus>::create_and_copy_from_slice(rnsbase);
        this->initialize();

    }

    void RNSBase::initialize() {

        size_t n = this->base_.size();
        
        Array<uint64_t> base_product(n, false);
        Array<uint64_t> punctured_product(n * n, false);
        Array<MultiplyUint64Operand> inv_punctured_product_mod_base(n, false);
        
        if (n == 1) {
            base_product[0] = this->base_[0].value();
            punctured_product[0] = 1;
            inv_punctured_product_mod_base[0] = MultiplyUint64Operand(1, this->base_[0]);
        } else {
            Array<uint64_t> base_values(n, false);
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
                    throw std::invalid_argument("RNSBase product is not invertible.");
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
            throw std::runtime_error("Invalid size of value.");
        }
        if (this->size() > 1) {
            Array<uint64_t> copied = Array<uint64_t>::create_and_copy_from_slice(value.as_const());
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

    __global__ void kernel_rnsbase_decompose_array(ConstSlice<Modulus> base, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index > result.size()) {
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

    void rnsbase_decompose_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        if (self.on_device() != from.on_device()) {
            throw std::invalid_argument("RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(from.size(), KERNEL_THREAD_COUNT);
            kernel_rnsbase_decompose_array<<<block_count, KERNEL_THREAD_COUNT>>>(self.base(), from, result);
        } else {
            host_rnsbase_decompose_array(self, from, result);
        }
    } 

    void RNSBase::decompose_array(Slice<uint64_t> value) const {
        if (value.size() % this->size() != 0) {
            throw std::invalid_argument("Value size must be a multiple of RNSBase size.");
        }
        if (this->size() == 1) {
            return; // nothing to do;
        }
        Array<uint64_t> cloned = Array<uint64_t>::create_and_copy_from_slice(value.as_const());
        rnsbase_decompose_array(*this, cloned.const_reference(), value);
    }

    void RNSBase::compose_single(Slice<uint64_t> value) const {
        if (value.size() != base_.size()) {
            throw std::runtime_error("Invalid size of value.");
        }
        if (this->size() > 1) {
            Array<uint64_t> copied = Array<uint64_t>::create_and_copy_from_slice(value.as_const());
            set_zero_uint(value);
            Array<uint64_t> temp_mpi(this->size(), false);
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
        if (global_index > count) {
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
        if (self.on_device() != from.on_device()) {
            throw std::invalid_argument("RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(from.size() / self.size(), KERNEL_THREAD_COUNT);
            kernel_rnsbase_compose_array<<<block_count, KERNEL_THREAD_COUNT>>>(
                self.base(), self.base_product(), self.punctured_product(), self.inv_punctured_product_mod_base(),
                from, result, temp_mpi
            );
        } else {
            host_rnsbase_compose_array(self, from, result, temp_mpi);
        }
    }

    void host_rnsbase_compose_rearrange_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        size_t n = self.size();
        size_t count = from.size() / n;
        ConstSlice<Modulus> base = self.base();
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
        if (global_index > count * n) {
            return;
        }
        size_t i = global_index / n;
        size_t j = global_index % n;
        result[i * n + j] = from[j * count + i];
    }

    void rnsbase_compose_rearrange_array(const RNSBase& self, ConstSlice<uint64_t> from, Slice<uint64_t> result) {
        if (self.on_device() != from.on_device()) {
            throw std::invalid_argument("RNSBase and value must be on the same memory.");
        }
        bool device = self.on_device();
        if (device) {
            size_t block_count = ceil_div<size_t>(from.size(), KERNEL_THREAD_COUNT);
            kernel_rnsbase_compose_rearrange_array<<<block_count, KERNEL_THREAD_COUNT>>>(self.size(), from, result);
        } else {
            host_rnsbase_compose_rearrange_array(self, from, result);
        }
    }

    void RNSBase::compose_array(Slice<uint64_t> value) const {
        if (value.size() % this->size() != 0) {
            throw std::invalid_argument("Value size must be a multiple of RNSBase size.");
        }
        if (this->size() == 1) {
            return; // nothing to do;
        }
        Array<uint64_t> cloned(value.size(), value.on_device());
        rnsbase_compose_rearrange_array(*this, value.as_const(), cloned.reference());
        Array<uint64_t> temp_mpi(value.size(), value.on_device());
        rnsbase_compose_array(*this, cloned.const_reference(), value, temp_mpi.reference());
    }

}}