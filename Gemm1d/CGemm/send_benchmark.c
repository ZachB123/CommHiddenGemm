#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char *argv[]) {

    int max_send = 30;

    if (argc == 2) {
        max_send = atoi(argv[1]);
    }

    printf("Max send is 2**%d\n", max_send);

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert(size >= 2);

    long send_buffer_size = 2;
    int iterations = 0;

    // was 1 << 30 MAKE THIS COMMAND LINE
    while (send_buffer_size <= (1 << max_send)) {

        double* data = malloc(send_buffer_size * sizeof(double));
        double* buffer = malloc(send_buffer_size * sizeof(double));

        double start_time = MPI_Wtime();
        if (rank == 0) {
            MPI_Send(data, send_buffer_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(buffer, send_buffer_size, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed_time = MPI_Wtime() - start_time;

        if (rank == 0) {
            char filename[100];
            sprintf(filename, "../benchmarks/c-n%d-sendbenchmark.csv", size);

            FILE *file = fopen(filename, "a");
            if (file == NULL) {
                printf("ERROR OPENING FILE\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fprintf(file, "%d,%ld,%f\n", size, send_buffer_size, elapsed_time);
            fclose(file);
        }

        free(data);
        free(buffer);

        iterations++;
        if (iterations >= 10) {
            send_buffer_size *= 2;
            iterations = 0;
        }
    }

    MPI_Finalize();
    return 0;
}