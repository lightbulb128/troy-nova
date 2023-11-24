#include <shared_mutex>
#include <mutex>
#include <vector>
#include "box.cuh"
#include "basics.cuh"
#include "number_theory.cuh"

namespace troy {namespace utils {

    const size_t GALOIS_GENERATOR = 3;

    class GaloisTool {

        bool device;
        size_t coeff_count_power_;
        size_t coeff_count_;
        
        mutable std::shared_mutex permutation_tables_rwlock;
        mutable std::vector<Array<size_t>> permutation_tables;

        static Array<size_t> generate_table_ntt(size_t coeff_count_power, size_t galois_element);

        void ensure_permutation_table(size_t galois_element) const;

    public:

        GaloisTool(const GaloisTool& other) = delete;
        GaloisTool(GaloisTool&& other) {
            this->device = other.device;
            this->coeff_count_power_ = other.coeff_count_power_;
            this->coeff_count_ = other.coeff_count_;
            this->permutation_tables = std::move(other.permutation_tables);
        }

        inline bool on_device() const noexcept { return device; }
        inline size_t coeff_count_power() const noexcept { return coeff_count_power_; }
        inline size_t coeff_count() const noexcept { return coeff_count_; }

        inline GaloisTool clone() const {
            GaloisTool cloned;
            cloned.device = this->device;
            cloned.coeff_count_power_ = this->coeff_count_power_;
            cloned.coeff_count_ = this->coeff_count_;
            cloned.permutation_tables = std::vector<Array<size_t>>(); permutation_tables.reserve(this->permutation_tables.size());
            for (size_t i = 0; i < this->permutation_tables.size(); i++) {
                cloned.permutation_tables.push_back(this->permutation_tables[i].clone());
            }
            return std::move(cloned);
        }

        inline void to_device_inplace() {
            if (device) {
                return;
            }
            device = true;
            for (size_t i = 0; i < permutation_tables.size(); i++) {
                permutation_tables[i].to_device_inplace();
            }
        }

        inline GaloisTool to_device() const {
            GaloisTool result = this->clone();
            result.to_device_inplace();
            return std::move(result);
        }
    
        inline GaloisTool() : device(false) {}
        
        GaloisTool(size_t coeff_count_power);

        size_t get_element_from_step(int step) const;
        inline std::vector<size_t> get_elements_from_steps(const std::vector<int>& steps) const {
            std::vector<size_t> result; result.reserve(steps.size());
            for (auto step : steps) {
                result.push_back(get_element_from_step(step));
            }
            return result;
        }

        std::vector<size_t> get_elements_all() const;

        /**
        Compute the index in the range of 0 to (coeff_count_ - 1) of a given Galois element.
        */
        inline static size_t get_index_from_element(size_t galois_element) {
            if (galois_element & 1 == 0) {
                throw std::invalid_argument("[GaloisTool::get_index_from_element] galois_element must be odd");
            }
            return (galois_element - 1) >> 1;
        }

        void apply_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) const;

        inline void apply_p(ConstSlice<uint64_t> poly, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) const {
            apply_ps(poly, 1, galois_element, moduli, result);
        }

        inline void apply(ConstSlice<uint64_t> component, size_t galois_element, ConstPointer<Modulus> modulus, Slice<uint64_t> result) const {
            apply_ps(component, 1, galois_element, ConstSlice<Modulus>::from_pointer(modulus), result);
        }

        void apply_ntt_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t galois_element, Slice<uint64_t> result) const;

        inline void apply_ntt_p(ConstSlice<uint64_t> poly, size_t coeff_modulus_size, size_t galois_element, Slice<uint64_t> result) const {
            apply_ntt_ps(poly, 1, coeff_modulus_size, galois_element, result);
        }

        inline void apply_ntt(ConstSlice<uint64_t> component, size_t galois_element, Slice<uint64_t> result) const {
            apply_ntt_ps(component, 1, 1, galois_element, result);
        }

        

    };

}}