# This script compares the performance of the host and device implementations of the Troy benchmark, with or without multithreading.

set -ex

cd build

make troybench -j64
./test/troybench -H -c 1 | tee host_1t.txt
./test/troybench -H -c 4 | tee host_4t.txt
./test/troybench -D -R 1000 -c 1 | tee device_1t.txt
./test/troybench -D -R 1000 -c 4 | tee device_4t1p.txt
./test/troybench -D -R 1000 -c 4 -mp | tee device_4t4p1d.txt
./test/troybench -D -R 1000 -c 4 -mp -md | tee device_4t4p4d.txt

python3 ../scripts/compare_bench.py host_1t.txt --other host_4t.txt > comp_host_1t_4t.txt
python3 ../scripts/compare_bench.py host_1t.txt --other device_1t.txt > comp_host_1t_device_1t.txt
python3 ../scripts/compare_bench.py host_1t.txt --other device_4t1p.txt > comp_host_1t_device_4t1p.txt
python3 ../scripts/compare_bench.py host_1t.txt --other device_4t4p1d.txt > comp_host_1t_device_4t4p1d.txt
python3 ../scripts/compare_bench.py host_1t.txt --other device_4t4p4d.txt > comp_host_1t_device_4t4p4d.txt
python3 ../scripts/compare_bench.py device_1t.txt --other device_4t1p.txt > comp_device_1t_4t.txt
python3 ../scripts/compare_bench.py device_4t1p.txt --other device_4t4p1d.txt > comp_device_1p_4p.txt
python3 ../scripts/compare_bench.py device_4t1p.txt --other device_4t4p4d.txt > comp_device_1p_4d.txt

cd ..