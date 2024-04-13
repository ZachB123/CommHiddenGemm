EXECS=gemm1d pingpong broadcastbenchmark

MPICC?=mpicc

all: ${EXECS}

gemm1d: gemm1d.c
	${MPICC} -pedantic -o gemm1d gemm1d.c

debug: gemm1d.c
	${MPICC} -g -o gemm1d gemm1d.c

pingpong: pingpong.c
	${MPICC} -pedantic -o pingpong pingpong.c

broadcastbenchmark: broadcast_benchmark.c
	${MPICC} -pedantic -o broadcastbenchmark broadcast_benchmark.c

clean:
	rm -f ${EXECS}


