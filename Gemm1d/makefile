EXECS=gemm1d

MPICC?=mpicc

all: ${EXECS}

gemm1d: gemm1d.c
	${MPICC} -pedantic -o gemm1d gemm1d.c

debug: gemm1d.c
	${MPICC} -g -o gemm1d gemm1d.c

clean:
	rm -f ${EXECS}
