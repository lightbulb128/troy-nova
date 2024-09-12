#pragma once
#include "box.h"
#include "box_batch.h"
#include "../modulus.h"
#include "memory_pool.h"
#include "uint_small_mod.h"

namespace troy {namespace utils {

    template <typename T>
    ConstSliceVec<T> slice_vec_to_const(const SliceVec<T>& slice_vec) {
        ConstSliceVec<T> result;
        result.reserve(slice_vec.size());
        for (auto& slice : slice_vec) result.push_back(slice);
        return result;
    }

    void copy_slice_b(const ConstSliceVec<uint64_t>& from, const SliceVec<uint64_t>& to, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    void copy_slice_b(const ConstSliceVec<uint8_t>& from, const SliceVec<uint8_t>& to, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    void set_slice_b(const uint64_t value, const SliceVec<uint64_t>& to, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    void modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void modulo_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    inline void modulo_p(ConstSlice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        modulo_ps(poly, 1, degree, moduli, result);
    }
    inline void modulo_bp(const ConstSliceVec<uint64_t>& polys, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        modulo_bps(polys, 1, degree, moduli, result, pool);
    }

    inline void modulo(ConstSlice<uint64_t> poly, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        modulo_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void modulo_b(const ConstSliceVec<uint64_t>& polys, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() != result.size()) throw std::invalid_argument("[modulo_b] polys.size() != result.size()");
        if (polys.size() > 0) modulo_bps(polys, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void modulo_inplace_ps(Slice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        modulo_ps(polys.as_const(), pcount, degree, moduli, polys);
    }
    inline void modulo_inplace_bps(const SliceVec<uint64_t>& polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        modulo_bps(slice_vec_to_const(polys), pcount, degree, moduli, polys, pool);
    }

    inline void modulo_inplace_p(Slice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli) {
        modulo_inplace_ps(poly, 1, degree, moduli);
    }
    inline void modulo_inplace_bp(const SliceVec<uint64_t>& polys, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        modulo_inplace_bps(polys, 1, degree, moduli, pool);
    }

    inline void modulo_inplace(Slice<uint64_t> poly, ConstPointer<Modulus> modulus) {
        modulo_inplace_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void modulo_inplace_b(const SliceVec<uint64_t>& polys, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() > 0) modulo_inplace_bps(polys, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }



    void negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void negate_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    
    inline void negate_p(ConstSlice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        negate_ps(poly, 1, degree, moduli, result);
    }
    inline void negate_bp(const ConstSliceVec<uint64_t>& polys, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negate_bps(polys, 1, degree, moduli, result, pool);
    }

    inline void negate(ConstSlice<uint64_t> poly, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        negate_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void negate_b(const ConstSliceVec<uint64_t>& polys, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() != result.size()) throw std::invalid_argument("[negate_b] polys.size() != result.size()");
        if (polys.size() > 0) negate_bps(polys, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void negate_inplace_ps(Slice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        negate_ps(polys.as_const(), pcount, degree, moduli, polys);
    }
    inline void negate_inplace_bps(const SliceVec<uint64_t>& polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negate_bps(slice_vec_to_const(polys), pcount, degree, moduli, polys, pool);
    }

    inline void negate_inplace_p(Slice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli) {
        negate_inplace_ps(poly, 1, degree, moduli);
    }
    inline void negate_inplace_bp(const SliceVec<uint64_t>& polys, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negate_inplace_bps(polys, 1, degree, moduli, pool);
    }

    inline void negate_inplace(Slice<uint64_t> poly, ConstPointer<Modulus> modulus) {
        negate_inplace_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void negate_inplace_b(const SliceVec<uint64_t>& polys, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() > 0) negate_inplace_bps(polys, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }

    void scatter_partial_ps(ConstSlice<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination);
    void scatter_partial_bps(const ConstSliceVec<uint64_t>& source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    inline void scatter_partial_p(ConstSlice<uint64_t> source_poly, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination) {
        scatter_partial_ps(source_poly, 1, source_degree, destination_degree, moduli_size, destination);
    }
    inline void scatter_partial_bp(const ConstSliceVec<uint64_t>& source_polys, size_t source_degree, size_t destination_degree, size_t moduli_size, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        scatter_partial_bps(source_polys, 1, source_degree, destination_degree, moduli_size, destination, pool);
    }
    inline void scatter_partial(ConstSlice<uint64_t> source_poly, size_t source_degree, size_t destination_degree, Slice<uint64_t> destination) {
        scatter_partial_ps(source_poly, 1, source_degree, destination_degree, 1, destination);
    }
    inline void scatter_partial_b(const ConstSliceVec<uint64_t>& source_polys, size_t source_degree, size_t destination_degree, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (source_polys.size() != destination.size()) throw std::invalid_argument("[scatter_partial_b] source_polys.size() != destination.size()");
        if (source_polys.size() > 0) scatter_partial_bps(source_polys, 1, source_degree, destination_degree, 1, destination, pool);
    }


    void add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void add_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    inline void add_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        add_ps(poly1, poly2, 1, degree, moduli, result);
    }
    inline void add_bp(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_bps(polys1, polys2, 1, degree, moduli, result, pool);
    }
    inline void add(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        add_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void add_b(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) throw std::invalid_argument("[add_b] polys1.size() != polys2.size() || polys1.size() != result.size()");
        if (polys1.size() > 0) add_bps(polys1, polys2, 1, polys1[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }

    inline void add_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        add_ps(polys1.as_const(), polys2, pcount, degree, moduli, polys1);
    }
    inline void add_inplace_bps(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_bps(slice_vec_to_const(polys1), polys2, pcount, degree, moduli, polys1, pool);
    }
    inline void add_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli) {
        add_inplace_ps(poly1, poly2, 1, degree, moduli);
    }
    inline void add_inplace_bp(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_inplace_bps(polys1, polys2, 1, degree, moduli, pool);
    }
    inline void add_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus) {
        add_inplace_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void add_inplace_b(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys1.size() != polys2.size()) throw std::invalid_argument("[add_inplace_b] polys1.size() != polys2.size()");
        if (polys1.size() > 0) add_inplace_bps(polys1, polys2, 1, polys1[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }


    void sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void sub_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    inline void sub_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        sub_ps(poly1, poly2, 1, degree, moduli, result);
    }
    inline void sub_bp(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_bps(polys1, polys2, 1, degree, moduli, result, pool);
    }
    inline void sub(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        sub_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void sub_b(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) throw std::invalid_argument("[sub_b] polys1.size() != polys2.size() || polys1.size() != result.size()");
        if (polys1.size() > 0) sub_bps(polys1, polys2, 1, polys1[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }

    inline void sub_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        sub_ps(polys1.as_const(), polys2, pcount, degree, moduli, polys1);
    }
    inline void sub_inplace_bps(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_bps(slice_vec_to_const(polys1), polys2, pcount, degree, moduli, polys1, pool);
    }
    inline void sub_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli) {
        sub_inplace_ps(poly1, poly2, 1, degree, moduli);
    }
    inline void sub_inplace_bp(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_inplace_bps(polys1, polys2, 1, degree, moduli, pool);
    }
    inline void sub_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus) {
        sub_inplace_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void sub_inplace_b(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys1.size() != polys2.size()) throw std::invalid_argument("[sub_inplace_b] polys1.size() != polys2.size()");
        if (polys1.size() > 0) sub_inplace_bps(polys1, polys2, 1, polys1[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }

    void add_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result);
    void add_partial_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    inline void add_partial_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        add_partial_ps(poly1, poly2, 1, degree1, degree2, moduli, result, degree_result);
    }
    inline void add_partial_bp(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_partial_bps(polys1, polys2, 1, degree1, degree2, moduli, result, degree_result, pool);
    }
    inline void add_partial(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, Slice<uint64_t> result, size_t degree_result) {
        add_partial_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), result, degree_result);
    }
    inline void add_partial_b(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_partial_bps(polys1, polys2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), result, degree_result, pool);
    }

    void sub_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result);
    void sub_partial_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    inline void sub_partial_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        sub_partial_ps(poly1, poly2, 1, degree1, degree2, moduli, result, degree_result);
    }
    inline void sub_partial_bp(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_partial_bps(polys1, polys2, 1, degree1, degree2, moduli, result, degree_result, pool);
    }
    inline void sub_partial(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, Slice<uint64_t> result, size_t degree_result) {
        sub_partial_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), result, degree_result);
    }
    inline void sub_partial_b(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_partial_bps(polys1, polys2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), result, degree_result, pool);
    }

    inline void add_partial_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        add_partial_ps(polys1.as_const(), polys2, pcount, degree1, degree2, moduli, polys1, degree1);
    }
    inline void add_partial_inplace_bps(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_partial_bps(slice_vec_to_const(polys1), polys2, pcount, degree1, degree2, moduli, polys1, degree1, pool);
    }
    inline void add_partial_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        add_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, moduli);
    }
    inline void add_partial_inplace_bp(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_partial_inplace_bps(polys1, polys2, 1, degree1, degree2, moduli, pool);
    }
    inline void add_partial_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus) {
        add_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void add_partial_inplace_b(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        add_partial_inplace_bps(polys1, polys2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), pool);
    }

    inline void sub_partial_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        sub_partial_ps(polys1.as_const(), polys2, pcount, degree1, degree2, moduli, polys1, degree1);
    }
    inline void sub_partial_inplace_bps(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_partial_bps(slice_vec_to_const(polys1), polys2, pcount, degree1, degree2, moduli, polys1, degree1, pool);
    }
    inline void sub_partial_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        sub_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, moduli);
    }
    inline void sub_partial_inplace_bp(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_partial_inplace_bps(polys1, polys2, 1, degree1, degree2, moduli, pool);
    }
    inline void sub_partial_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus) {
        sub_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void sub_partial_inplace_b(const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        sub_partial_inplace_bps(polys1, polys2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), pool);
    }

    void add_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);

    inline void add_scalar_p(ConstSlice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        add_scalar_ps(poly, scalar, 1, degree, moduli, result);
    }

    inline void add_scalar(ConstSlice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        add_scalar_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void add_scalar_inplace_ps(Slice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        add_scalar_ps(polys.as_const(), scalar, pcount, degree, moduli, polys);
    }

    inline void add_scalar_inplace_p(Slice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli) {
        add_scalar_inplace_ps(poly, scalar, 1, degree, moduli);
    }

    inline void add_scalar_inplace(Slice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus) {
        add_scalar_inplace_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }



    void sub_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);

    inline void sub_scalar_p(ConstSlice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        sub_scalar_ps(poly, scalar, 1, degree, moduli, result);
    }

    inline void sub_scalar(ConstSlice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        sub_scalar_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void sub_scalar_inplace_ps(Slice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        sub_scalar_ps(polys.as_const(), scalar, pcount, degree, moduli, polys);
    }

    inline void sub_scalar_inplace_p(Slice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli) {
        sub_scalar_inplace_ps(poly, scalar, 1, degree, moduli);
    }

    inline void sub_scalar_inplace(Slice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus) {
        sub_scalar_inplace_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }




    void multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void multiply_scalar_bps(const ConstSliceVec<uint64_t>& polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    inline void multiply_scalar_p(ConstSlice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        multiply_scalar_ps(poly, scalar, 1, degree, moduli, result);
    }
    inline void multiply_scalar_bp(const ConstSliceVec<uint64_t>& polys, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        multiply_scalar_bps(polys, scalar, 1, degree, moduli, result, pool);
    }

    inline void multiply_scalar(ConstSlice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        multiply_scalar_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void multiply_scalar_b(const ConstSliceVec<uint64_t>& polys, uint64_t scalar, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() != result.size()) throw std::invalid_argument("[multiply_scalar_b] polys.size() != result.size()");
        if (polys.size() > 0) multiply_scalar_bps(polys, scalar, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void multiply_scalar_inplace_ps(Slice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_scalar_ps(polys.as_const(), scalar, pcount, degree, moduli, polys);
    }
    inline void multiply_scalar_inplace_bps(const SliceVec<uint64_t>& polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        ConstSliceVec<uint64_t> polys_const; polys_const.reserve(polys.size()); 
        for (auto& poly : polys) polys_const.push_back(poly);
        multiply_scalar_bps(polys_const, scalar, pcount, degree, moduli, polys, pool);
    }

    inline void multiply_scalar_inplace_p(Slice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_scalar_inplace_ps(poly, scalar, 1, degree, moduli);
    }
    inline void multiply_scalar_inplace_bp(const SliceVec<uint64_t>& polys, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        multiply_scalar_inplace_bps(polys, scalar, 1, degree, moduli, pool);
    }

    inline void multiply_scalar_inplace(Slice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus) {
        multiply_scalar_inplace_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void multiply_scalar_inplace_b(const SliceVec<uint64_t>& polys, uint64_t scalar, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() > 0) multiply_scalar_inplace_bps(polys, scalar, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }


    void multiply_scalars_ps(ConstSlice<uint64_t> polys, ConstSlice<uint64_t> scalars, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);

    inline void multiply_scalars_p(ConstSlice<uint64_t> poly, ConstSlice<uint64_t> scalars, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        multiply_scalars_ps(poly, scalars, 1, degree, moduli, result);
    }



    inline void multiply_scalars_inplace_ps(Slice<uint64_t> polys, ConstSlice<uint64_t> scalars, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_scalars_ps(polys.as_const(), scalars, pcount, degree, moduli, polys);
    }

    inline void multiply_scalars_inplace_p(Slice<uint64_t> poly, ConstSlice<uint64_t> scalars, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_scalars_inplace_ps(poly, scalars, 1, degree, moduli);
    }

    

    void multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void multiply_uint64operand_bps(
        const ConstSliceVec<uint64_t>& polys, ConstSlice<MultiplyUint64Operand> operand, 
        size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    inline void multiply_uint64operand_p(ConstSlice<uint64_t> poly, ConstSlice<MultiplyUint64Operand> operand, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        multiply_uint64operand_ps(poly, operand, 1, degree, moduli, result);
    }
    inline void multiply_uint64operand_bp(
        const ConstSliceVec<uint64_t>& polys, ConstSlice<MultiplyUint64Operand> operand, 
        size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        multiply_uint64operand_bps(polys, operand, 1, degree, moduli, result, pool);
    }

    inline void multiply_uint64operand(ConstSlice<uint64_t> poly, ConstPointer<MultiplyUint64Operand> operand, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        multiply_uint64operand_ps(poly, ConstSlice<MultiplyUint64Operand>::from_pointer(operand), 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void multiply_uint64operand_b(
        const ConstSliceVec<uint64_t>& polys, ConstPointer<MultiplyUint64Operand> operand, 
        ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        if (polys.size() != result.size()) throw std::invalid_argument("[multiply_uint64operand_b] polys.size() != result.size()");
        if (polys.size() > 0) multiply_uint64operand_bps(polys, ConstSlice<MultiplyUint64Operand>::from_pointer(operand), 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void multiply_uint64operand_inplace_ps(Slice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_uint64operand_ps(polys.as_const(), operand, pcount, degree, moduli, polys);
    }
    inline void multiply_uint64operand_inplace_bps(
        const SliceVec<uint64_t>& polys, ConstSlice<MultiplyUint64Operand> operand, 
        size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        ConstSliceVec<uint64_t> polys_const = rcollect_as_const(polys);
        multiply_uint64operand_bps(polys_const, operand, pcount, degree, moduli, polys, pool);
    }

    inline void multiply_uint64operand_inplace_p(Slice<uint64_t> poly, ConstSlice<MultiplyUint64Operand> operand, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_uint64operand_inplace_ps(poly, operand, 1, degree, moduli);
    }
    inline void multiply_uint64operand_inplace_bp(
        const SliceVec<uint64_t>& polys, ConstSlice<MultiplyUint64Operand> operand, 
        size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        multiply_uint64operand_inplace_bps(polys, operand, 1, degree, moduli, pool);
    }

    inline void multiply_uint64operand_inplace(Slice<uint64_t> poly, ConstPointer<MultiplyUint64Operand> operand, ConstPointer<Modulus> modulus) {
        multiply_uint64operand_inplace_ps(poly, ConstSlice<MultiplyUint64Operand>::from_pointer(operand), 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }
    inline void multiply_uint64operand_inplace_b(
        const SliceVec<uint64_t>& polys, ConstPointer<MultiplyUint64Operand> operand, 
        ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        if (polys.size() > 0) multiply_uint64operand_inplace_bps(polys, ConstSlice<MultiplyUint64Operand>::from_pointer(operand), 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }



    void dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);

    void dyadic_product_bps(
        const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, 
        size_t pcount, size_t degree, ConstSlice<Modulus> moduli, 
        const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    inline void dyadic_product_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        dyadic_product_ps(poly1, poly2, 1, degree, moduli, result);
    }

    inline void dyadic_product_bp(
        const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, 
        size_t degree, ConstSlice<Modulus> moduli, 
        const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        dyadic_product_bps(polys1, polys2, 1, degree, moduli, result, pool);
    }

    inline void dyadic_product(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        dyadic_product_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }

    inline void dyadic_product_b(
        const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, 
        ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) throw std::invalid_argument("[dyadic_product_b] polys1.size() != polys2.size() || polys1.size() != result.size()");
        if (polys1.size() > 0) dyadic_product_bps(polys1, polys2, 1, polys1[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void dyadic_product_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        dyadic_product_ps(polys1.as_const(), polys2, pcount, degree, moduli, polys1);
    }

    inline void dyadic_product_inplace_bps(
        const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, 
        size_t pcount, size_t degree, ConstSlice<Modulus> moduli, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        dyadic_product_bps(slice_vec_to_const(polys1), polys2, pcount, degree, moduli, polys1, pool);
    }

    inline void dyadic_product_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli) {
        dyadic_product_inplace_ps(poly1, poly2, 1, degree, moduli);
    }

    inline void dyadic_product_inplace_bp(
        const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, 
        size_t degree, ConstSlice<Modulus> moduli, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        dyadic_product_inplace_bps(polys1, polys2, 1, degree, moduli, pool);
    }

    inline void dyadic_product_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus) {
        dyadic_product_inplace_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }

    inline void dyadic_product_inplace_b(
        const SliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, 
        ConstPointer<Modulus> modulus, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        if (polys1.size() != polys2.size()) throw std::invalid_argument("[dyadic_product_inplace_b] polys1.size() != polys2.size()");
        if (polys1.size() > 0) dyadic_product_inplace_bps(polys1, polys2, 1, polys1[0].size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }



    void negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    void negacyclic_shift_bps(const ConstSliceVec<uint64_t>& polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    inline void negacyclic_shift_p(ConstSlice<uint64_t> poly, size_t shift, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        negacyclic_shift_ps(poly, shift, 1, degree, moduli, result);
    }
    inline void negacyclic_shift_bp(const ConstSliceVec<uint64_t>& polys, size_t shift, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_shift_bps(polys, shift, 1, degree, moduli, result, pool);
    }

    inline void negacyclic_shift(ConstSlice<uint64_t> poly, size_t shift, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        negacyclic_shift_ps(poly, shift, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }
    inline void negacyclic_shift_b(const ConstSliceVec<uint64_t>& polys, size_t shift, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (polys.size() != result.size()) throw std::invalid_argument("[negacyclic_shift_b] polys.size() != result.size()");
        if (polys.size() > 0) negacyclic_shift_bps(polys, shift, 1, polys[0].size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void negacyclic_multiply_mononomial_ps(ConstSlice<uint64_t> polys, uint64_t mono_coeff, uint64_t mono_exponent, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        Array<uint64_t> temp(result.size(), result.on_device(), pool);
        multiply_scalar_ps(polys, mono_coeff, pcount, degree, moduli, temp.reference());
        negacyclic_shift_ps(temp.const_reference(), mono_exponent, pcount, degree, moduli, result);
    }

    inline void negacyclic_multiply_mononomial_p(ConstSlice<uint64_t> poly, uint64_t mono_coeff, uint64_t mono_exponent, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomial_ps(poly, mono_coeff, mono_exponent, 1, degree, moduli, result, pool);
    }

    inline void negacyclic_multiply_mononomial(ConstSlice<uint64_t> poly, uint64_t mono_coeff, uint64_t mono_exponent, ConstPointer<Modulus> modulus, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomial_ps(poly, mono_coeff, mono_exponent, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result, pool);
    }



    inline void negacyclic_multiply_mononomial_inplace_ps(Slice<uint64_t> polys, uint64_t mono_coeff, uint64_t mono_exponent, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomial_ps(polys.as_const(), mono_coeff, mono_exponent, pcount, degree, moduli, polys, pool);
    }

    inline void negacyclic_multiply_mononomial_inplace_p(Slice<uint64_t> poly, uint64_t mono_coeff, uint64_t mono_exponent, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomial_inplace_ps(poly, mono_coeff, mono_exponent, 1, degree, moduli, pool);
    }

    inline void negacyclic_multiply_mononomial_inplace(Slice<uint64_t> poly, uint64_t mono_coeff, uint64_t mono_exponent, ConstPointer<Modulus> modulus, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomial_inplace_ps(poly, mono_coeff, mono_exponent, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), pool);
    }


    inline void negacyclic_multiply_mononomials_ps(ConstSlice<uint64_t> polys, ConstSlice<uint64_t> mono_coeffs, uint64_t mono_exponent, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        Array<uint64_t> temp(result.size(), result.on_device(), pool);
        multiply_scalars_ps(polys, mono_coeffs, pcount, degree, moduli, temp.reference());
        negacyclic_shift_ps(temp.const_reference(), mono_exponent, pcount, degree, moduli, result);
    }

    inline void negacyclic_multiply_mononomials_p(ConstSlice<uint64_t> poly, ConstSlice<uint64_t> mono_coeffs, uint64_t mono_exponent, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomials_ps(poly, mono_coeffs, mono_exponent, 1, degree, moduli, result, pool);
    }

    inline void negacyclic_multiply_mononomials_inplace_ps(Slice<uint64_t> polys, ConstSlice<uint64_t> mono_coeffs, uint64_t mono_exponent, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomials_ps(polys.as_const(), mono_coeffs, mono_exponent, pcount, degree, moduli, polys, pool);
    }

    inline void negacyclic_multiply_mononomials_inplace_p(Slice<uint64_t> poly, ConstSlice<uint64_t> mono_coeffs, uint64_t mono_exponent, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        negacyclic_multiply_mononomials_inplace_ps(poly, mono_coeffs, mono_exponent, 1, degree, moduli, pool);
    }

}}