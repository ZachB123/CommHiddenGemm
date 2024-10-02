import sys

GEMM_NODES = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128]
BENCHMARK_NODES = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48]


def write_sbatch(
    file_path,
    job_name,
    account,
    nodes,
    ntasks_per_node,
    time,
    partition,
    out,
    email,
    mem=0,
    *args,
):
    # file path does not include .sbatch
    # *args is just a list of other things to write to the file
    with open(f"{file_path}.sbatch", "w") as file:
        file.write("#!/usr/bin/env bash\n")
        file.write(f"#SBATCH -J{job_name}\n")
        file.write(f"#SBATCH --account={account}\n")
        file.write(f"#SBATCH --nodes={nodes}\n")
        file.write(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        file.write(f"#SBATCH --mem={mem}\n")
        file.write(f"#SBATCH -t{time}\n")
        file.write(f"#SBATCH -p{partition}\n")
        file.write(f"#SBATCH -o{out}-%j.out\n")
        file.write(f"#SBATCH --mail-type=BEGIN,END,FAIL\n")
        file.write(f"#SBATCH --mail-user={email}\n")

        for arg in args:
            file.write(arg + "\n")


def write_gemm():
    for nodes in GEMM_NODES:
        time = "5-00:00:00"
        if nodes > 8:
            time = "96:00:00"
            if nodes > 32:
                time = "60:00:00"
        for ntasks in [1, 2]:
            write_sbatch(
                f"gemm-N{nodes}-n{ntasks}",
                f"Gemm-{nodes}:{ntasks}",
                "hive-rvuduc3",
                nodes,
                ntasks,
                time,
                "hive",
                f"Gemm-{nodes}:{ntasks}",
                "zbuchholz3@gatech.edu",
                0,
                'echo "Started on `/bin/hostname`"',
                "module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn",
                "pip install -r ../../requirements.txt",
                f"srun python driver.py {ntasks}",
            )


# def write_pingpong_benchmarks():
#     write_sbatch(
#         f"python_pingpong",  # path
#         f"PingPongPython",  # job name
#         "hive-rvuduc3",
#         2,
#         1,
#         "24:00:00",
#         "hive",
#         f"PingPongPython",
#         "zbuchholz3@gatech.edu",
#         0,
#         'echo "Started on `/bin/hostname`"',
#         "module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn",
#         "srun python pingpong.py 31 1",
#     )
#     write_sbatch(
#         f"c_pingpong",  # path
#         f"PingPongC",  # job name
#         "hive-rvuduc3",
#         2,
#         1,
#         "24:00:00",
#         "hive",
#         f"PingPongC",
#         "zbuchholz3@gatech.edu",
#         0,
#         'echo "Started on `/bin/hostname`"',
#         "make clean",
#         "make",
#         "srun ./pingpong 31 1",
#     )


def write_pingpong_benchmarks():
    write_sbatch(
        f"pingpong",
        f"pingpong_python_and_C",
        "hive-rvuduc3",
        2,
        1,
        "24:00:00",
        f"PingPongPythonC",
        "hive",
        "zbuchholz3@gatech.edu",
        0,
        'echo "Started on `/bin/hostname`"',
        "module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn",
        "srun python pingpong.py 31 1",
        "make",
        "srun ./pingpong 31 1",
    )


def write_broadcast_benchmarks():
    time = "24:00:00"
    for nodes in BENCHMARK_NODES:
        for ntasks in [1, 2]:
            for program in ["python", "c"]:
                if program == "python":
                    build = [
                        "module load anaconda3 py-mpi4py/3.1.2-mva2-rzdjbn",
                        f"srun python broadcast_benchmark.py 31 {ntasks}",
                    ]
                else:
                    build = [
                        "make clean",
                        "make",
                        f"srun ./broadcastbenchmark 31 {ntasks}",
                    ]
                write_sbatch(
                    f"{program}-broadcast-N{nodes}-n{ntasks}",  # path
                    f"{program}-broadcast-{nodes}:{ntasks}",  # job name
                    "hive-rvuduc3",
                    nodes,
                    ntasks,
                    time,
                    "hive",
                    f"{program}-broadcast-{nodes}:{ntasks}",
                    "zbuchholz3@gatech.edu",
                    0,
                    'echo "Started on `/bin/hostname`"',
                    *build,
                )


def write_basic_benchmarks():
    write_pingpong_benchmarks()
    write_broadcast_benchmarks()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "b":
        write_basic_benchmarks()
    else:
        write_gemm()


if __name__ == "__main__":
    main()
    # write_pingpong_benchmarks()
