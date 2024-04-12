EXECS=gemm1d sendbenchmark broadcastbenchmark

MPICC?=mpicc

all: ${EXECS}

gemm1d: gemm1d.c
	${MPICC} -pedantic -o gemm1d gemm1d.c

debug: gemm1d.c
	${MPICC} -g -o gemm1d gemm1d.c

sendbenchmark: send_benchmark.c
	${MPICC} -pedantic -o sendbenchmark send_benchmark.c

broadcastbenchmark: broadcast_benchmark.c
	${MPICC} -pedantic -o broadcastbenchmark broadcast_benchmark.c

clean:
	rm -f ${EXECS}


