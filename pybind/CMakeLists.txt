pybind11_add_module(pytroy binder.cu)
target_link_libraries(pytroy PRIVATE troy)

# copy "test.py" to build directory
configure_file(test.py test.py COPYONLY)