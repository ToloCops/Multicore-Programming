#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void serialMatrixMult(
        double* A  /* in  */,
        double* B  /* in  */,
        double* C  /* out */,
        int n      /* in  */) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    int n = atoi(argv[1]);
    double *A, *B, *C, *local_A, *local_C;

    if (rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        C = (double*)malloc(n * n * sizeof(double));

        srand(time(0));
        for (int i = 0; i < n * n; i++) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }
    } else {
        B = (double*)malloc(n * n * sizeof(double));
    }

    int rows_per_proc = n / size;
    local_A = (double*)malloc(rows_per_proc * n * sizeof(double));
    local_C = (double*)malloc(rows_per_proc * n * sizeof(double));

    MPI_Scatter(A, rows_per_proc * n, MPI_DOUBLE, local_A, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < n; ++j) {
            local_C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_proc * n, MPI_DOUBLE, C, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double* seq_C = (double*)malloc(n * n * sizeof(double));
        serialMatrixMult(A, B, seq_C, n);

        int correct = 1;
        for(int i = 0; i < n * n; ++i) {
            if (C[i] != seq_C[i]) {
                correct = 0;
                break;
            }
        }

        if (correct) {
            printf("Matrix multiplication is correct!\n");
        }
        else {
            printf("Matrix multiplication is incorrect\n");
        }

        free(seq_C);
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;    
}