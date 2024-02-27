EXECS=gemm1d

MPICC?=mpicc

all: ${EXECS}

gemm1d: gemm1d.c
	${MPICC} -o gemm1d gemm1d.c 

clean:
	rm -f ${EXECS}
