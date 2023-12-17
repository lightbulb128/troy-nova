from setuptools import setup
import os

data_files = ['*.so', "__init__.py"]

# is there a __init__.pyi file?
if os.path.isfile('./__init__.pyi'):
    data_files.append('py.typed')
    data_files.append('__init__.pyi')

print("DATA_FILES", data_files)

setup(
    name = "pytroy",
    version = "0.0.1",
    description = "pytroy: GPU implementation of BFV, CKKS and BGV LHE",
    author = "Lightbulb",
    author_email = "lxq22@mails.tsinghua.edu.cn",
    packages = ['pytroy'],
    package_dir = {'pytroy': '.'},
    package_data = {'pytroy': data_files},
)