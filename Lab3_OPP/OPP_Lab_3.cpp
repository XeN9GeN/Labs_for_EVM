#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <iostream>
int main(int argc, char* argv[]) {
    int size, rank;//size-число процессов
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n1, n2, n3;
    if (argc == 2) {
        n1 = atoi(argv[1]);
        n2 = atoi(argv[1]);
        n3 = atoi(argv[1]);
    }
    int side_sq = (int)sqrt(size);

    int row = rank / side_sq;//координаты данного процесса в решётке процессов
    int col = rank % side_sq;

    //колво строк/столбов из матрицы для конкретной строки/столба процессов в сетке
    int rows_A = n1 / side_sq;
    int cols_B = n3 / side_sq;


    double* A_full = NULL;
    double* B_full = NULL;
    if (rank == 0) {
        A_full = (double*)malloc(n1 * n2 * sizeof(double));
        B_full = (double*)malloc(n2 * n3 * sizeof(double));

        for (int i = 0; i < n1; i++) {
            for (int k = 0; k < n2; k++) {
                A_full[i * n2 + k] = (double)(i + 1);
            }
        }
        //C[i] = (i+1)(j+1)*n2
        //i==j: n2(i+1)^2
        for (int k = 0; k < n2; k++) {
            for (int j = 0; j < n3; j++) {
                B_full[k * n3 + j] = (double)(j + 1);
            }
        }
    }

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);


    //элементы нескольких строк/столбов матрицы А/B на 1 процесс
    double* A_strip = (double*)malloc(rows_A * n2 * sizeof(double));
    double* B_strip = (double*)malloc(n2 * cols_B * sizeof(double));
    double* C_block = (double*)calloc(rows_A * cols_B, sizeof(double));
    double* C_full  = (double*)calloc(n1 * n3, sizeof(double));


    if (col == 0) {
        MPI_Scatter(A_full, rows_A * n2, MPI_DOUBLE, A_strip, rows_A * n2, MPI_DOUBLE, 0, col_comm);
    }
    MPI_Bcast(A_strip, rows_A* n2, MPI_DOUBLE, 0, row_comm);

    if (row == 0) {
        MPI_Datatype col_type_B = MPI_DATATYPE_NULL;
        MPI_Datatype col_stripe_resized = MPI_DATATYPE_NULL;
        MPI_Type_vector(n2, cols_B, n3, MPI_DOUBLE, &col_type_B);
        MPI_Type_create_resized(col_type_B, 0, cols_B * sizeof(double), &col_stripe_resized);
        MPI_Type_commit(&col_stripe_resized);

        MPI_Scatter(B_full, 1, col_stripe_resized, B_strip, n2 * cols_B, MPI_DOUBLE, 0, row_comm);

        MPI_Type_free(&col_stripe_resized);
        MPI_Type_free(&col_type_B);
    }
    MPI_Bcast(B_strip, n2* cols_B, MPI_DOUBLE, 0, col_comm);




    double start_time = MPI_Wtime();
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            for (int k = 0; k < n2; k++) {
                //A_strip - полоска в горизонталь из строк, элемент это i-е колво строк * ширину матрицы + сдвиг на нужный элемент в i-й строке
                //B_strip - элем * ширина вертикальной полоски + сдвиг на j
                //C_block[] - пропускаем i полных строк длиной cols_B и сдвиг на j
                C_block[i * cols_B + j] += A_strip[i * n2 + k] * B_strip[k * cols_B + j];
            }
        }
    }
    double end_time = MPI_Wtime();




    int* counts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            counts[i] = 1;//один объект для каждого процесса
            int i_row = i / side_sq;
            int i_cols = i % side_sq;
            //первая скобка - прыжок по вертикали, потом по горизонтали
            displs[i] = (i_row * rows_A * n3) + (i_cols * cols_B);
            //индекс в C_full, с которой должен начаться блоки данные от процесса i
        }
    }

    MPI_Datatype block_type, resized_block_type;
    int big_size[2] = {n1, n3};
    int sub_size[2] = {rows_A, cols_B};
    int start_indices[2] = {0, 0};

    MPI_Type_create_subarray(2, big_size, sub_size, start_indices, MPI_ORDER_C, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    if (rank == 0) {
        C_full = (double*)malloc(n1 * n3 * sizeof(double));
    }
    MPI_Gatherv(C_block, rows_A * cols_B, MPI_DOUBLE, C_full, counts, displs, resized_block_type, 0, MPI_COMM_WORLD);




    if (rank == 0) {
        int wrong_eq = 0;
        double delta = 1e-9;

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n3; j++) {
                double expected = (double)(i + 1) * (j + 1) * n2;
                double actual = C_full[i * n3 + j];

                if (fabs(actual - expected) > delta) {
                    if (wrong_eq < 5) {
                        printf("Error in C[%d][%d]: получено %f, ожидалось %f\n", i, j, actual, expected);
                    }
                    wrong_eq++;
                }
            }
        }
        if (wrong_eq == 0) {
            printf("Good\n");
        }


        free(C_full);
        free(counts);
        free(displs);
    }
    MPI_Type_free(&resized_block_type);

    if (rank == 0) {
        printf("Time: %f seconds\n", end_time - start_time);
    }

    free(A_strip);
    free(B_strip);
    free(C_block);
    if (rank == 0) {
        free(A_full);
        free(B_full);
    }


    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();

    return 0;
}
