#pragma once
#include "../modulus.cuh"
#include "uint_small_mod.cuh"
#include "number_theory.cuh"

namespace troy {namespace utils {

    class RNSBase {
    
        Array<Modulus> base_;
        Array<uint64_t> base_product_;
        // n * n
        Array<uint64_t> punctured_product_;
        Array<MultiplyUint64Operand> inv_punctured_product_mod_base_;
        bool device;

        void initialize();

    public:

        inline RNSBase(): device(false) {}

        RNSBase(ConstSlice<Modulus> rnsbase);

        __host__ __device__
        bool on_device() const { return device; }

        __host__ __device__ 
        inline ConstSlice<Modulus> base() const {
            return base_.const_reference();
        }

        __host__ __device__
        inline ConstSlice<uint64_t> base_product() const {
            return base_product_.const_reference();
        }

        __host__ __device__
        inline ConstSlice<uint64_t> punctured_product(size_t index) const {
            return punctured_product_
                .const_slice(index * base_.size(), (index + 1) * base_.size());
        }

        __host__ __device__
        inline ConstSlice<uint64_t> punctured_product() const {
            return punctured_product_.const_reference();
        }

        __host__ __device__
        inline ConstSlice<MultiplyUint64Operand> inv_punctured_product_mod_base() const {
            return inv_punctured_product_mod_base_.const_reference();
        }

        __host__ __device__ size_t size() const { return base_.size(); }

        __host__ __device__
        inline bool contains(const Modulus& modulus) const {
            for (size_t i = 0; i < base_.size(); ++i) {
                if (base_[i].value() == modulus.value()) {
                    return true;
                }
            }
            return false;
        }

        __host__ __device__
        inline bool is_subbase_of(const RNSBase& other) const {
            for (size_t i = 0; i < base_.size(); ++i) {
                if (!other.contains(base_[i])) {
                    return false;
                }
            }
            return true;
        }

        __host__ __device__
        inline bool is_superbase_of(const RNSBase& other) const {
            return other.is_subbase_of(*this);
        }

        __host__ __device__
        inline bool is_proper_subbase_of(const RNSBase& other) const {
            return is_subbase_of(other) && this->size() < other.size();
        }

        __host__ __device__
        inline bool is_proper_superbase_of(const RNSBase& other) const {
            return is_superbase_of(other) && this->size() > other.size();
        }

        inline RNSBase extend_modulus(const Modulus& modulus) {
            if (this->base_.on_device()) {
                throw std::runtime_error("[RNSBase::extend_modulus] Cannot extend RNSBase from device memory.");
            }
            Array<Modulus> new_base(base_.size() + 1, false);
            for (size_t i = 0; i < base_.size(); ++i) {
                new_base[i] = base_[i];
            }
            new_base[base_.size()] = modulus;
            return RNSBase(new_base.const_reference());
        }

        /*
        inline RNSBase extend(const RNSBase& other) {
            if (this->base_.on_device() || other.base_.on_device()) {
                throw std::runtime_error("Cannot extend RNSBase from device memory.");
            }
            Array<Modulus> new_base(base_.size() + other.base_.size(), false);
            for (size_t i = 0; i < base_.size(); ++i) {
                new_base[i] = base_[i];
            }
            for (size_t i = 0; i < other.base_.size(); ++i) {
                new_base[base_.size() + i] = other.base_[i];
            }
            return RNSBase(new_base.const_reference());
        }
        */

        void decompose_single(Slice<uint64_t> value) const;

        void decompose_array(Slice<uint64_t> values) const;

        void compose_single(Slice<uint64_t> value) const;

        void compose_array(Slice<uint64_t> values) const;

        inline RNSBase clone() const {
            RNSBase cloned;
            cloned.base_ = base_.clone();
            cloned.base_product_ = base_product_.clone();
            cloned.punctured_product_ = punctured_product_.clone();
            cloned.inv_punctured_product_mod_base_ = inv_punctured_product_mod_base_.clone();
            cloned.device = device;
            return cloned;
        }

        inline void to_device_inplace() {
            if (device) {
                return;
            }
            base_.to_device_inplace();
            base_product_.to_device_inplace();
            punctured_product_.to_device_inplace();
            inv_punctured_product_mod_base_.to_device_inplace();
            device = true;
        }

        inline RNSBase to_device() {
            RNSBase rnsbase = this->clone();
            rnsbase.to_device_inplace();
            return rnsbase;
        }
        

    };

    class BaseConverter {

        bool device;
        RNSBase ibase;
        RNSBase obase;

        // o.len() * i.len()
        Array<uint64_t> base_change_matrix_;

    public:

        __host__ __device__ inline bool on_device() const { return device; }
        
        const RNSBase& input_base() const { return ibase; }
        const RNSBase& output_base() const { return obase; }
        ConstSlice<uint64_t> base_change_matrix() const { return base_change_matrix_.const_reference(); }

        inline BaseConverter(): device(false) {}

        inline BaseConverter(const RNSBase& ibase, const RNSBase& obase) {
            if (ibase.on_device() || obase.on_device()) {
                throw std::runtime_error("[BaseConverter::BaseConverter] Cannot create BaseConverter from device memory.");
            }
            this->ibase = ibase.clone();
            this->obase = obase.clone();
            this->device = false;
            this->initialize();
        }

        inline void initialize() {
            Array<uint64_t> base_change_matrix = Array<uint64_t>(
                obase.size() * ibase.size(), false);
            for (size_t i = 0; i < obase.size(); ++i) {
                for (size_t j = 0; j < ibase.size(); ++j) {
                    base_change_matrix[i * ibase.size() + j] = 
                        utils::modulo_uint(
                            this->ibase.punctured_product(j),
                            this->obase.base()[i]
                        );
                }
            }
            this->base_change_matrix_ = std::move(base_change_matrix);
        }

        inline BaseConverter clone() const {
            BaseConverter converter;
            converter.ibase = ibase.clone();
            converter.obase = obase.clone();
            converter.base_change_matrix_ = base_change_matrix_.clone();
            converter.device = device;
            return converter;
        }

        inline void to_device_inplace() {
            if (device) {
                return;
            }
            ibase.to_device_inplace();
            obase.to_device_inplace();
            base_change_matrix_.to_device_inplace();
            device = true;
        }

        inline BaseConverter to_device() const {
            BaseConverter converter = this->clone();
            converter.to_device_inplace();
            return converter;
        }
        
        void fast_convert_array(ConstSlice<uint64_t> input, Slice<uint64_t> output) const;
        
        // See "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme" (CT-RSA 2019) for details
        void exact_convey_array(ConstSlice<uint64_t> input, Slice<uint64_t> output) const;


    };

}}
