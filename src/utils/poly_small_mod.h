#pragma once
#include "box.h"
#include "../modulus.h"
#include "uint_small_mod.h"

namespace troy {namespace utils {

    void modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    
    inline void modulo_p(ConstSlice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        modulo_ps(poly, 1, degree, moduli, result);
    }

    inline void modulo(ConstSlice<uint64_t> poly, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        modulo_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void modulo_inplace_ps(Slice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        modulo_ps(polys.as_const(), pcount, degree, moduli, polys);
    }

    inline void modulo_inplace_p(Slice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli) {
        modulo_inplace_ps(poly, 1, degree, moduli);
    }

    inline void modulo_inplace(Slice<uint64_t> poly, ConstPointer<Modulus> modulus) {
        modulo_inplace_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }



    void negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    
    inline void negate_p(ConstSlice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        negate_ps(poly, 1, degree, moduli, result);
    }

    inline void negate(ConstSlice<uint64_t> poly, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        negate_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void negate_inplace_ps(Slice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        negate_ps(polys.as_const(), pcount, degree, moduli, polys);
    }

    inline void negate_inplace_p(Slice<uint64_t> poly, size_t degree, ConstSlice<Modulus> moduli) {
        negate_inplace_ps(poly, 1, degree, moduli);
    }

    inline void negate_inplace(Slice<uint64_t> poly, ConstPointer<Modulus> modulus) {
        negate_inplace_ps(poly, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }

    void scatter_partial_ps(ConstSlice<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination);
    inline void scatter_partial_p(ConstSlice<uint64_t> source_poly, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination) {
        scatter_partial_ps(source_poly, 1, source_degree, destination_degree, moduli_size, destination);
    }
    inline void scatter_partial(ConstSlice<uint64_t> source_poly, size_t source_degree, size_t destination_degree, Slice<uint64_t> destination) {
        scatter_partial_ps(source_poly, 1, source_degree, destination_degree, 1, destination);
    }


    void add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    inline void add_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        add_ps(poly1, poly2, 1, degree, moduli, result);
    }
    inline void add(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        add_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }

    inline void add_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        add_ps(polys1.as_const(), polys2, pcount, degree, moduli, polys1);
    }
    inline void add_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli) {
        add_inplace_ps(poly1, poly2, 1, degree, moduli);
    }
    inline void add_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus) {
        add_inplace_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }


    void sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);
    inline void sub_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        sub_ps(poly1, poly2, 1, degree, moduli, result);
    }
    inline void sub(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        sub_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }

    inline void sub_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        sub_ps(polys1.as_const(), polys2, pcount, degree, moduli, polys1);
    }
    inline void sub_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli) {
        sub_inplace_ps(poly1, poly2, 1, degree, moduli);
    }
    inline void sub_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus) {
        sub_inplace_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }

    void add_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result);
    inline void add_partial_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        add_partial_ps(poly1, poly2, 1, degree1, degree2, moduli, result, degree_result);
    }
    inline void add_partial(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, Slice<uint64_t> result, size_t degree_result) {
        add_partial_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), result, degree_result);
    }

    void sub_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result);
    inline void sub_partial_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        sub_partial_ps(poly1, poly2, 1, degree1, degree2, moduli, result, degree_result);
    }
    inline void sub_partial(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus, Slice<uint64_t> result, size_t degree_result) {
        sub_partial_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus), result, degree_result);
    }

    inline void add_partial_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        add_partial_ps(polys1.as_const(), polys2, pcount, degree1, degree2, moduli, polys1, degree1);
    }
    inline void add_partial_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        add_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, moduli);
    }
    inline void add_partial_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus) {
        add_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus));
    }

    inline void sub_partial_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        sub_partial_ps(polys1.as_const(), polys2, pcount, degree1, degree2, moduli, polys1, degree1);
    }
    inline void sub_partial_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli) {
        sub_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, moduli);
    }
    inline void sub_partial_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree1, size_t degree2, ConstPointer<Modulus> modulus) {
        sub_partial_inplace_ps(poly1, poly2, 1, degree1, degree2, ConstSlice<Modulus>::from_pointer(modulus));
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

    inline void multiply_scalar_p(ConstSlice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        multiply_scalar_ps(poly, scalar, 1, degree, moduli, result);
    }

    inline void multiply_scalar(ConstSlice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        multiply_scalar_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void multiply_scalar_inplace_ps(Slice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_scalar_ps(polys.as_const(), scalar, pcount, degree, moduli, polys);
    }

    inline void multiply_scalar_inplace_p(Slice<uint64_t> poly, uint64_t scalar, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_scalar_inplace_ps(poly, scalar, 1, degree, moduli);
    }

    inline void multiply_scalar_inplace(Slice<uint64_t> poly, uint64_t scalar, ConstPointer<Modulus> modulus) {
        multiply_scalar_inplace_ps(poly, scalar, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
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

    inline void multiply_uint64operand_p(ConstSlice<uint64_t> poly, ConstSlice<MultiplyUint64Operand> operand, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        multiply_uint64operand_ps(poly, operand, 1, degree, moduli, result);
    }

    inline void multiply_uint64operand(ConstSlice<uint64_t> poly, ConstPointer<MultiplyUint64Operand> operand, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        multiply_uint64operand_ps(poly, ConstSlice<MultiplyUint64Operand>::from_pointer(operand), 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void multiply_uint64operand_inplace_ps(Slice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_uint64operand_ps(polys.as_const(), operand, pcount, degree, moduli, polys);
    }

    inline void multiply_uint64operand_inplace_p(Slice<uint64_t> poly, ConstSlice<MultiplyUint64Operand> operand, size_t degree, ConstSlice<Modulus> moduli) {
        multiply_uint64operand_inplace_ps(poly, operand, 1, degree, moduli);
    }

    inline void multiply_uint64operand_inplace(Slice<uint64_t> poly, ConstPointer<MultiplyUint64Operand> operand, ConstPointer<Modulus> modulus) {
        multiply_uint64operand_inplace_ps(poly, ConstSlice<MultiplyUint64Operand>::from_pointer(operand), 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }



    void dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);

    inline void dyadic_product_p(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        dyadic_product_ps(poly1, poly2, 1, degree, moduli, result);
    }

    inline void dyadic_product(ConstSlice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        dyadic_product_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
    }



    inline void dyadic_product_inplace_ps(Slice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli) {
        dyadic_product_ps(polys1.as_const(), polys2, pcount, degree, moduli, polys1);
    }

    inline void dyadic_product_inplace_p(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, size_t degree, ConstSlice<Modulus> moduli) {
        dyadic_product_inplace_ps(poly1, poly2, 1, degree, moduli);
    }

    inline void dyadic_product_inplace(Slice<uint64_t> poly1, ConstSlice<uint64_t> poly2, ConstPointer<Modulus> modulus) {
        dyadic_product_inplace_ps(poly1, poly2, 1, poly1.size(), ConstSlice<Modulus>::from_pointer(modulus));
    }



    void negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result);

    inline void negacyclic_shift_p(ConstSlice<uint64_t> poly, size_t shift, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        negacyclic_shift_ps(poly, shift, 1, degree, moduli, result);
    }

    inline void negacyclic_shift(ConstSlice<uint64_t> poly, size_t shift, ConstPointer<Modulus> modulus, Slice<uint64_t> result) {
        negacyclic_shift_ps(poly, shift, 1, poly.size(), ConstSlice<Modulus>::from_pointer(modulus), result);
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