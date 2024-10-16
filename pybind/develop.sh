# exit when error
set -ex

cd ../build
cmake .. -DTROY_PYBIND=ON -DCMAKE_BUILD_TYPE=Release
make pytroy_raw -j64

# get filename called "pybind/pytroy_raw.cpython*"
filename=$(ls pybind/pytroy_raw.cpython*)

# copy it to pybind folder
cp $filename ../pybind

# return to the pybind folder
cd ../pybind

# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install stubgen
pip install wheel pybind11-stubgen numpy typing-extensions

# if there is __init__.pyi, remove it
if [ -f __init__.pyi ]; then
    rm __init__.pyi
fi

# build wheels
pip install -e .

# run stubgen
pybind11-stubgen pytroy_raw

# move and rename the stub file
mv stubs/pytroy_raw.pyi __init__.pyi

# remove the stubs folder
rm -r stubs

# regenerate the wheel
python setup.py sdist bdist_wheel
rm -r pytroy.egg-info

# reinstall the wheel
pip install --force-reinstall dist/pytroy*.whl

rm pytroy_raw.cpython*.so

python tests/test.py