#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

/*
AG A COL 1d hidden gemm

C = C + A*B

A = m x k
B = k x n
C = m x n

most efficient if n < k < m

for testing purposes I have created 2 examples

1st is a mini version
m = 16, k = 8, n = 4

2nd is larger m = 60, k = 30, n = 15

all matrix values are random in the rand [-10,9)

*/


/*
Questions: should the matrices be floats?
should I make it so only rank 0 has access to the matrices at the start and it has to scatter them
*/

#define MINI_M 16
#define MINI_K 8
#define MINI_N 4

#define LARGE_M 60
#define LARGE_K 30
#define LARGE_N 15

const int MINI_MATRIX_A[MINI_M][MINI_K] = {
    {-5, -10,  5,  -3,   6, -10,   5, -8},
    {-4,  -2, -4, -10,   6,   0,  -7,  2},
    {-10,  1,  3,  -4,   2,  -7,  -7, -6},
    {-9,   3,  8, -10,   5,  -7,   5,  7},
    {-9,  -4,  2,   1,   6,   6,  -1, -1},
    {8,    4,  0,   3,   7,  -1,   5,  5},
    {-2,   8,  0,  -6, -10,   9,  -2, -5},
    {-4, -10, -1,  -6,   5,  -2,   3, -2},
    {-6,   6,  6,  -4,   2,   5,   0,  6},
    {-10, -9, -9,  -6,   8,  -3,   1, -8},
    {6,    8,  6,  -5,  -9,  -1,  -4,  1},
    {-7,  -6,  4,   1,  -5,   8, -10, -8},
    {-5,   9, -7,  -8,  -1,   1,   0, -8},
    {-8,   9, -2,   1,  -1,   5,   6, -5},
    {-2,   4, -5,   3,  -3,  -8,   2,  2},
    {-7,   4,  2, -10,  -1,   3,  -5,  5},
};

const int MINI_MATRIX_B[MINI_K][MINI_N] = {
    {0, 8, 8, 5},
    {-7, 4, 6, 3},
    {-7, -5, 5, -7},
    {-3, 5, 8, -3},
    {8, 9, -5, 9},
    {5, 2, 5, 9},
    {-2, 1, 5, 7},
    {6, -6, 2, -2},
};


// C is just set to 0 now but would work if it was set to something else
int MINI_MATRIX_C[MINI_M][MINI_N] = {
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
    {0,0,0,0},
};

const int EXPECTED_MINI_MATRIX_C[MINI_M][MINI_N] = {
    {-16, -33, -170, -66}, 
    {146, -35, -205, 33},
    {-57, -78, -183, -138},
    {-10, -156, -115, -59},
    {85, -22, -85, 29},
    {34, 131, 107, 122},
    {-99, -58, 59, 19}, 
    {107, -41, -169, 27},
    {5, -82, 13, 9},
    {143, 14, -293, 72}, 
    {-146, -68, 108, -93}, 
    {-17, -86, -65, -111}, 
    {-41, 32, -91, 91},
    {-77, 24, 38, 86},
    {-58, -13, -4, -61},
    {35, -138, -97, -34}
};

const int LARGE_MATRIX_A[LARGE_M][LARGE_K] = {};
const int LARGE_MATRIX_B[LARGE_K][LARGE_N] = {};
const int LARGE_MATRIX_C[LARGE_M][LARGE_N] = {};

int rank;
int size;

void print_matrix(int m, int n, const int M[m][n]) {
    // this will probably break if 4 digit numbers or greater are introduced
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d, ", M[i][j]);
        }
        printf("\n");
    }
}

void print_buffer(int m, int n, const int* M) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // why not sizeof int
            printf("%4d, ", M[i * n + j]);//*(M + (n * i + j)));
        }
        printf("\n");
    }
}

void standard_matrix_multiply(int m, int k, int n, const int A[m][k], const int B[k][n], int C[m][n]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k_prime = 0; k_prime < k; k_prime++) {
                C[i][j] = C[i][j] + A[i][k_prime] * B[k_prime][j];
            }
        }
    }
}

// columns at [start, end)
// we only return int* here because we have compressed the matrix into a 1d object
// user must free this
int* split_matrix_along_columns(const int start_col, const int end_col, const int rows, const int columns, const int M[rows][columns]) {
    // we malloc
    int num_cols = end_col - start_col;
    int* sub_matrix = (int*) malloc(rows * num_cols * sizeof(int));
    
    for (int i = 0; i < rows; i++) {
        for (int j = start_col; j < end_col; j++) {
            sub_matrix[i * num_cols + j - start_col] = M[i][j];
        }
    }

    return sub_matrix;
}

int main(int argc, int* argv) {

    {
        volatile int i = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
            sleep(5);
    }


    // for mini matrix the world size must be 4
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // make sure we run on 4 processors so everything splits nice
    assert(size == 4);
    // initial distributions

    // note the MINI_X we use is the dimension we have split the matrix across
    int* A_I = split_matrix_along_columns(rank * (MINI_K / size), (rank + 1) * (MINI_K / size), MINI_M, MINI_K, MINI_MATRIX_A);
    int* B_I = split_matrix_along_columns(rank * (MINI_N / size), (rank + 1) * (MINI_N / size), MINI_K, MINI_N, MINI_MATRIX_B);
    int* C_I = split_matrix_along_columns(rank * (MINI_N / size), (rank + 1) * (MINI_N / size), MINI_M, MINI_N, MINI_MATRIX_C);

    int A_I_cols = (rank + 1) * (MINI_K / size) - rank * (MINI_K / size);
    int B_I_cols = (rank + 1) * (MINI_N / size) - rank * (MINI_N / size);
    int C_I_cols = (rank + 1) * (MINI_N / size) - rank * (MINI_N / size);


    if (rank == -1) {
        print_buffer(MINI_M, 2, A_I);
        printf("\n");
        print_buffer(MINI_K, 1, B_I);
        printf("\n");
        print_buffer(MINI_M, 1, C_I);
    }


    int prev_rank = (rank + size - 1) % size;
    int next_rank = (rank + 1) % size;
    int A_elements = MINI_M * (((rank + 1) * (MINI_K / size)) - (rank * (MINI_K / size)));

    for (int cycle = 0; cycle < size; cycle++) {

        int* A_temp = (int*) malloc(A_elements * sizeof(int));

        MPI_Request send_request, recv_request;
        MPI_Status recv_status;

        MPI_Isend(A_I, A_elements, MPI_INT, next_rank, 0, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(A_temp, A_elements, MPI_INT, prev_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);

        // printf("%d is doing computation now with %d elements\n", rank, A_elements);
        
        // if (rank == 0) {
        //     print_buffer(MINI_M, 2, A_I);
        //     printf("\n");
        // }

        // need a B start

        int shared_width = MINI_K / size;


        for (int i = 0; i < MINI_M; i++) {
            int B_start_index = (shared_width * (rank - cycle)) % MINI_K;
            for (int j = B_start_index; j < B_start_index + shared_width; j++) {
                for (int k = 0; k < shared_width; k++) {
                    if (rank == 1) {
                        printf("(i,j,k) = (%d,%d,%d)\n", i, j, k);
                    }
                    // int test = C_I[i * C_I_cols + j];
                    // C_I[i * C_I_cols + j] = C_I[i * C_I_cols + j] + A_I[i * A_I_cols + k] * B_I[k * B_I_cols + j];
                }
            }
        }

        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request, &recv_status);
        
        free(A_I);
        A_I = A_temp;
    }


    if (rank == 0) {
        // print_buffer(16, 4, C_I);
    }

    // standard_matrix_multiply(MINI_M, MINI_K, MINI_N, MINI_MATRIX_A, MINI_MATRIX_B, MINI_MATRIX_C);
    // print_matrix(MINI_M, MINI_N, MINI_MATRIX_C);

    free(A_I);
    free(B_I);
    free(C_I);

    MPI_Finalize();

    return 0;
}