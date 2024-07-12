#include <gtest/gtest.h>
#include "../test.h"
#include "../../src/utils/box.h"

using namespace troy::utils;

class Foo {
    int num;
public:
    Foo() : num(0) {}
    Foo(int num) : num(num) {}
    int get_num() const { return num; }
    void set_num(int num) { this->num = num; }
};

namespace box {

    TEST(Box, ItCompiles) {
        EXPECT_EQ(1, 1);
        utils::MemoryPool::Destroy();
    }

    TEST(Box, HostBox) {

        int x = 1;
        Box<int> x_box(new int(x), false, nullptr);
        EXPECT_EQ(*x_box, 1);

        *x_box = 2;
        EXPECT_EQ(*x_box, 2);

        Foo f(12);
        Box<Foo> f_box(new Foo(f), false, nullptr);
        EXPECT_EQ(f_box->get_num(), 12);

        f_box->set_num(13);
        EXPECT_EQ(f_box->get_num(), 13);

        ConstPointer<Foo> f_const_ptr = f_box.as_const_pointer();
        EXPECT_EQ(f_const_ptr->get_num(), 13);

        Pointer<Foo> f_ptr = f_box.as_pointer();
        EXPECT_EQ(f_ptr->get_num(), 13);
        f_ptr->set_num(14);
        EXPECT_EQ(f_ptr->get_num(), 14);
        EXPECT_EQ(f_box->get_num(), 14);
        utils::MemoryPool::Destroy();

    }

}
