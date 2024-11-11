#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(void) {
    int comm_sz, my_rank;   // Number of processes and rank of each process
    int darts_per_proc;     // Processes split the darts bewtwen them

    double square_side = 2.0;
    int circle_radius  = 1;
    int darts_num      = 1000000;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    darts_per_proc = darts_num / comm_sz;

    srand(time(0) + my_rank);  // Seed for random number generator based on time and rank
    int local_count = 0;       // Number of darts that fall inside the circle

    if (my_rank != 0) {
        for (int i = 0; i < darts_per_proc; i++) {
            double x = (double)rand() / RAND_MAX * square_side - 1.0;  // Random x coordinate bewtween -1 and 1
            double y = (double)rand() / RAND_MAX * square_side - 1.0;
            if (x * x + y * y <= circle_radius * circle_radius) {      // Check if dart falls inside the circle
                local_count++;
            }
        }
        MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);      // Send the local count to the root process
    }
    else {
        for (int i = 0; i < darts_per_proc; i++) {                     // Root also processes darts
            double x = (double)rand() / RAND_MAX * square_side - 1.0;
            double y = (double)rand() / RAND_MAX * square_side - 1.0;
            if (x * x + y * y <= circle_radius * circle_radius) {
                local_count++;
            }
        }
        int temp;  // Used to store the local count of each process
        for (int i = 1; i < comm_sz; i++) {
            MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_count += temp;
        }
        double pi = 4 * (double)local_count / darts_num;  // Uses the proportion number in circle : total number of darts = pi : 4
        printf("Estimated pi: %f\n", pi);
    }

    MPI_Finalize();
}