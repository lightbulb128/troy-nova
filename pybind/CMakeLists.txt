pybind11_add_module(pytroy_raw binder.cu)
target_link_libraries(pytroy_raw PRIVATE troy)

# copy "test.py" to build directory
configure_file(test.py test.py COPYONLY)