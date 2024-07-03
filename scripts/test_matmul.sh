set -ex

cd build/test

make bench_matmul -j64

M=100
R=105
N=110

function run_bench {
    local extra_args=$1
    ./bench_matmul --bfv         -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --ckks        -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --bgv         -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --bfv -rt 32  -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --bfv -rt 64  -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --bfv -rt 128 -m ${M} -r ${R} -n ${N} ${extra_args}
}

function run_bench_no_ring2k {
    local extra_args=$1
    ./bench_matmul --bfv         -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --ckks        -m ${M} -r ${R} -n ${N} ${extra_args}
    ./bench_matmul --bgv         -m ${M} -r ${R} -n ${N} ${extra_args}
}

run_bench "--host --threads 1"
run_bench "--host --threads 4"
run_bench "--host --threads 4 --no-pack-lwes"
run_bench "--host --threads 4 --mod-switch-down-levels 1"

run_bench "--device --threads 1"
run_bench "--device --threads 4"
run_bench "--device --threads 4 --no-pack-lwes"
run_bench "--device --threads 4 --mod-switch-down-levels 1"

run_bench "--device --multiple-devices --threads 4"
run_bench "--device --multiple-devices --threads 4 --no-pack-lwes"
run_bench "--device --multiple-devices --threads 4 --mod-switch-down-levels 1"

cd ../..