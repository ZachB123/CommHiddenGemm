#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>

#define NUM_TRIALS 100

int main(int argc, char *argv[]) {

    int max_send = 30;
    int num_cores_per_node = -1;

    if (argc >= 2) {
        max_send = atoi(argv[1]);
    }
        if (argc == 3) {
        num_cores_per_node = atoi(argv[2]);
    }

    printf("Max send is 2**%d\n", max_send);

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert(size >= 2);

    long send_buffer_size = 1024;


    // was 1 << 30 MAKE THIS COMMAND LINE
    while (send_buffer_size <= (1 << max_send)) {
        // printf("max: %d, curr: %ld\n", (1 << max_send), send_buffer_size);
        for (int _ = 0; _ < NUM_TRIALS; _++) {

            double* data = malloc(send_buffer_size * sizeof(double));
            if (data == NULL) {
                printf("MALLOC FAILURE");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            double* buffer = malloc(send_buffer_size * sizeof(double));
            if (buffer == NULL) {
                free(data);
                printf("MALLOC FAILURE");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            double start_time = MPI_Wtime();
            if (rank == 0) {
                MPI_Send(data, send_buffer_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Recv(buffer, send_buffer_size, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(data, send_buffer_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            if (rank == 0) {
                MPI_Recv(buffer, send_buffer_size, MPI_DOUBLE, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            double elapsed_time = MPI_Wtime() - start_time;
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                char filename[150];
                sprintf(filename, "../../BasicBenchmarks/basic-benchmarks/c-N%d-n%d-sendbenchmark.csv", size, num_cores_per_node);

                FILE *file = fopen(filename, "a");
                if (file == NULL) {
                    free(data);
                    free(buffer);
                    printf("ERROR OPENING FILE\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                fprintf(file, "%d,%ld,%f\n", size, 2 * send_buffer_size, elapsed_time);
                fclose(file);
            }

            free(data);
            free(buffer);

        }
        send_buffer_size *= 2;
    }

    MPI_Finalize();
    return 0;
}