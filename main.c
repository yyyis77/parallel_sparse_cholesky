// Created by Zhichao Hu & Yongyi Yang
// Creation time: 1th April, 2017
// Project Name: Sparse Cholesky Factorization In Parallel Using OpenMPI
// Project Description: Realized the Cholesky Factorization on sparse matrix,
//                      columns in the matrix could be computed simultaneously,
//                      if there is no dependency between them.

// The matrix is passed into the function as a parameter
// The answer can be compared to the serial algotirhm by passing in the second parameter
// ./main <matrix.txt> <answer.txt>

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MATRIX_SIZE 500

double **mat_in;
int **mat_res_verifier;

struct node_info {
    int col_no;
    int tier_level;
    int dependency_count;
    int dependency_col[MATRIX_SIZE];
};

struct tier_map {
    int col_no;
    int tier_level_origin;
};

struct send_map {
    int col_no;
    int target_procs[MATRIX_SIZE];
    int target_count;
};

int **tiers;
int current_tier;
int current_tier_size;
int zero_tier_size;

struct node_info *all_columns;
struct node_info *all_columns_orig;
struct tier_map *all_columns_sortmap;
struct send_map *self_send_map;
int **self_recv_map;

int **rank_col_map;
int *iteration_per_rank;
int rank_id, num_proc;

// Function prototypes
int check_sat(struct node_info node, int tier);
void dependency_checker(struct tier_map *all_nodes_sorted, int num_proc);
void quick_sort(struct tier_map *all_nodes_sorted, int low, int high);
int partition(struct tier_map *all_nodes_sortmap, int low, int high);

void cdiv(double **matrix, int col_num_i);

void cmod(double **matrix, int col_num_j, int col_num_k);


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int i, j, k;
    double start_timer, end_timer, timer_period;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Request request;

    mat_in = malloc(sizeof(double *) * MATRIX_SIZE);
    mat_res_verifier = malloc(sizeof(int *) * MATRIX_SIZE);
    tiers = malloc(sizeof(int *) * MATRIX_SIZE);
    for (i = 0; i < MATRIX_SIZE; i++) {
        mat_in[i] = malloc(sizeof(double) * MATRIX_SIZE);
        mat_res_verifier[i] = malloc(sizeof(int) * MATRIX_SIZE);
        tiers[i] = malloc(sizeof(int) * MATRIX_SIZE);
    }

    // Record the start time of the program
    if (rank_id == 0) {
        start_timer = clock();
    }

    // Read in the matrix for factorization
    FILE *fp1 = fopen(argv[1], "r");
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            fscanf(fp1, "%lf", &mat_in[i][j]);
        }
    }
    fclose(fp1);

    // Read in the matrix for comparison
    fp1 = fopen(argv[2], "r");
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            fscanf(fp1, "%d", &mat_res_verifier[i][j]);
        }
    }
    fclose(fp1);

    zero_tier_size = 0;

    // Initialize variables and tables for dependency check
    struct node_info *current_node;
    all_columns = malloc(sizeof(struct node_info) * MATRIX_SIZE);
    all_columns_orig = malloc(sizeof(struct node_info) * MATRIX_SIZE);
    all_columns_sortmap = malloc(sizeof(struct tier_map) * MATRIX_SIZE);

    // zero_tier_size is already set here.
    dependency_checker(all_columns_sortmap, num_proc);

    // Buffer with the same size as the input matrix
    // Used for receiving columns from corresponding processors
    double **buffer_mat = (double **) malloc(sizeof(double *) * MATRIX_SIZE);
    for (i = 0; i < MATRIX_SIZE; i++) {
        buffer_mat[i] = (double *) malloc(sizeof(double *) * MATRIX_SIZE);
    }

    // Record the time spend on preprocessing
    printf("Pre-processing complete\n");
    if (rank_id == 0) {
        end_timer = clock();
        timer_period = (end_timer - start_timer) / CLOCKS_PER_SEC;
        printf("Pre-processing spent: %lf seconds\n", timer_period);
        start_timer = clock();
    }

    MPI_Barrier(MPI_COMM_WORLD);


    //==================== construct iteration_per_rank ====================//
    int total_iteration = MATRIX_SIZE / num_proc;
    if (rank_id < MATRIX_SIZE % num_proc) {
        total_iteration++;
    }

    if (rank_id == 0) {
        iteration_per_rank = (int *) malloc(sizeof(int) * num_proc);
        // processor 0 has the full information about the interation per processor
        iteration_per_rank[0] = total_iteration;
        for (i = 1; i < num_proc; i++) {
            MPI_Recv(&iteration_per_rank[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }

    } else {
        MPI_Send(&total_iteration, 1, MPI_INT, 0, rank_id, MPI_COMM_WORLD);
    }

    //==================== construct rank_col_map ====================//
    int temp;
    int *self_cols;

    if (rank_id == 0) {
        rank_col_map = (int **) malloc(sizeof(int *) * num_proc);
        for (temp = 0; temp < num_proc; temp++) {
            rank_col_map[temp] = (int *) malloc(sizeof(int) * iteration_per_rank[temp]);
        }
        self_cols = (int *) malloc(sizeof(int) * total_iteration);

        j = 0;
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (i % num_proc == 0) {
                self_cols[j] = all_columns[i].col_no;
                j++;
            }
        }

        memcpy(rank_col_map[0], self_cols, sizeof(int) * total_iteration);
        for (temp = 1; temp < num_proc; temp++) {
            MPI_Recv(rank_col_map[temp], iteration_per_rank[temp], MPI_INT, temp, temp, MPI_COMM_WORLD,
                     MPI_STATUSES_IGNORE);
        }
    } else {
        self_cols = (int *) malloc(sizeof(int) * total_iteration);
        j = 0;
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (i % num_proc == rank_id) {
                self_cols[j] = all_columns[i].col_no;
                j++;
            }
        }
        MPI_Isend(self_cols, total_iteration, MPI_INT, 0, rank_id, MPI_COMM_WORLD, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==================== construct self_send_map & self_recv_map ====================//
    int iter;
    int recv_col;
    int send_count;
    int send_rank[MATRIX_SIZE];

    self_send_map = malloc(sizeof(struct send_map) * total_iteration);
    self_recv_map = malloc(sizeof(int *) * total_iteration);
    for (i = 0; i < total_iteration; i++) {
        self_recv_map[i] = malloc(sizeof(int) * MATRIX_SIZE);
    }

    // Construct self_send_map
    // Record the number of procesor which need to send current data to
    for (iter = 0; iter < total_iteration; iter++) {
        k = 0;
        current_node = &all_columns_orig[self_cols[iter]];
        for (i = current_node->col_no; i < MATRIX_SIZE; i++) {
            for (j = 0; j < all_columns[i].dependency_count; j++) {
                if (all_columns[i].dependency_col[j] == current_node->col_no) {
                    if (rank_id == i % num_proc) {
                        continue;
                    }
                    send_rank[k] = i % num_proc;
                    self_send_map[iter].target_procs[k] = send_rank[k];
                    //printf("rank %d computing %d send to rank %d to compute %d col\n",rank_id,current_node->col_no,send_rank[k],all_columns[i].col_no);
                    k++;
                    break;
                }
            }
        }

        send_count = k;
        self_send_map[iter].target_count = send_count;
        for (i = 0; i < self_send_map[iter].target_count; i++) {
            j = i + 1;
            while (j < self_send_map[iter].target_count) {
                if (self_send_map[iter].target_procs[j] == self_send_map[iter].target_procs[i]) {
                    for (k = j; k < self_send_map[iter].target_count; k++) {
                        self_send_map[iter].target_procs[k] = self_send_map[iter].target_procs[k + 1];
                    }
                    self_send_map[iter].target_count--;
                } else
                    j++;
            }
        }

    }

    // Construct self_recv_map
    // Record the number of procesor which need to receive data from
    for (iter = 0; iter < total_iteration; iter++) {
        k = 0;
        current_node = &all_columns_orig[self_cols[iter]];
        for (i = 0; i < current_node->dependency_count; i++) {
            recv_col = current_node->dependency_col[i];
            for (j = 0; j < MATRIX_SIZE; j++) {
                if (all_columns[j].col_no == recv_col) {
                    self_recv_map[iter][k] = j % num_proc;
                    k++;
                    break;
                }
            }
        }
    }

    //==================== computation ====================//
    // Record whether the column has been computed
    int available[MATRIX_SIZE] = {0};

    for (iter = 0; iter < total_iteration; iter++) {
        current_node = &all_columns_orig[self_cols[iter]];
        // The first tier column has no dependency
        // Just call cdiv
        if (current_node->tier_level == 0) {
            cdiv(mat_in, current_node->col_no);
            available[current_node->col_no] = 1;

        } else {
            // Receive and cmod with each dependent column
            // available[col] is set to 1 once this column is the lateset version
            // Latest means received this column in previous iteration
            //              OR computation complete on this processor
            for (i = 0; i < current_node->dependency_count; i++) {
                recv_col = current_node->dependency_col[i];
                // Check the self_recv_map to get corresponding processor's number
                int recv_rank;
                recv_rank = self_recv_map[iter][i];

                if (rank_id == recv_rank && available[recv_col] == 1) {
                    cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);

                } else if (available[recv_col] == 0) {
                    MPI_Irecv(buffer_mat[recv_col],
                              MATRIX_SIZE,
                              MPI_DOUBLE,
                              recv_rank,
                              recv_col,
                              MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUSES_IGNORE);
                    available[recv_col] = 1;

                    int ccc;
                    for (ccc = 0; ccc < MATRIX_SIZE; ccc++) {
                        mat_in[ccc][recv_col] = buffer_mat[recv_col][ccc];
                    }

                    cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);

                } else {
                    cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);
                }
            }

            // After cmod with all dependent column, cdiv itself
            cdiv(mat_in, current_node->col_no);
            available[current_node->col_no] = 1;
        }

        // check the send map and send data to corresponding processor
        for (i = 0; i < MATRIX_SIZE; i++) {
            buffer_mat[current_node->col_no][i] = mat_in[i][current_node->col_no];
        }

        for (i = 0; i < self_send_map[iter].target_count; i++) {
            MPI_Isend(buffer_mat[current_node->col_no],
                      MATRIX_SIZE,
                      MPI_DOUBLE,
                    //send_rank[i],
                      self_send_map[iter].target_procs[i],
                      current_node->col_no,
                      MPI_COMM_WORLD, &request);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //==================== generate final result ====================//
    int ccc;
    if (rank_id == 0) {
        for (i = 1; i < num_proc; i++) {
            for (j = 0; j < iteration_per_rank[i]; j++) {
                recv_col = rank_col_map[i][j];
                MPI_Recv(buffer_mat[recv_col], MATRIX_SIZE, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                for (ccc = 0; ccc < MATRIX_SIZE; ccc++) {
                    mat_in[ccc][recv_col] = buffer_mat[recv_col][ccc];
                }
            }
        }
    } else {
        for (i = 0; i < total_iteration; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                buffer_mat[self_cols[i]][j] = mat_in[j][self_cols[i]];
            }
            MPI_Send(buffer_mat[self_cols[i]], MATRIX_SIZE, MPI_DOUBLE, 0, rank_id, MPI_COMM_WORLD);
        }
    }

    // Record the computation time
    if (rank_id == 0) {
        end_timer = clock();
        timer_period = (end_timer - start_timer) / CLOCKS_PER_SEC;
        printf("Computation spent: %lf seconds\n", timer_period);
    }

    //==================== check the correctness of final result ====================//
    if (rank_id == 0) {
        int error_flag = 0;
        for (i = 0; i < MATRIX_SIZE; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                if (i >= j) {
                    if (mat_in[i][j] != mat_res_verifier[i][j]) {
                        printf("Wrong anawer.\n");
                        error_flag = 1;
                        break;
                    }
                }
            }
            if (error_flag == 1)
                break;
        }
        if (error_flag == 0) {
            printf("Test passed.\n");
        }
    }


    MPI_Finalize();
}

void cmod(double **matrix, int col_num_j, int col_num_k) {
    int i, j, k;
    j = col_num_j;
    k = col_num_k;

    for (i = j; i < MATRIX_SIZE; i++) {
        matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[j][k];
    }
}

void cdiv(double **matrix, int col_num_j) {
    int i, j;
    j = col_num_j;
    matrix[col_num_j][col_num_j] = sqrt(matrix[col_num_j][col_num_j]);

    for (i = j + 1; i < MATRIX_SIZE; i++) {
        matrix[i][j] = matrix[i][j] / matrix[j][j];
    }
}

int check_sat(struct node_info node, int tier) {
    int i, j, k;
    int dep_count;
    int found_flag = 0;

    for (dep_count = 0; dep_count < node.dependency_count; dep_count++) {
        found_flag = 0;
        for (i = tier - 1; i >= 0; i--) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                if (tiers[i][j] == node.dependency_col[dep_count]) {
                    found_flag = 1;
                    break;
                }
            }
            if (found_flag == 1)
                break;
        }
        if (found_flag == 0)
            return 0;
    }

    return 1;
}

void dependency_checker(struct tier_map *all_nodes_sortmap, int num_proc) {
    int row_counter1;
    int col_counter1, col_counter2;

    int i, j;
    int cols_left = MATRIX_SIZE;

    int **fill_in_temp = malloc(sizeof(int *) * MATRIX_SIZE);
    for (i = 0; i < MATRIX_SIZE; i++)
        fill_in_temp[i] = malloc(sizeof(int) * MATRIX_SIZE);


    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            if (mat_in[i][j] == 0)
                fill_in_temp[i][j] = 0;
            else
                fill_in_temp[i][j] = 1;
        }
    }

    for (col_counter1 = 1; col_counter1 < MATRIX_SIZE; col_counter1++) {
        for (col_counter2 = 0; col_counter2 < col_counter1; col_counter2++) {
            if (mat_in[col_counter1][col_counter2] != 0) {
                for (row_counter1 = col_counter1; row_counter1 < MATRIX_SIZE; row_counter1++) {
                    if (mat_in[row_counter1][col_counter2] != 0 && fill_in_temp[row_counter1][col_counter1] == 0)
                        fill_in_temp[row_counter1][col_counter1] = 2;
                }
            }
        }
    }

    all_columns[0].col_no = 0;
    all_columns[0].dependency_count = 0;
    all_columns[0].tier_level = -1;

    for (col_counter1 = 1; col_counter1 < MATRIX_SIZE; col_counter1++) {
        all_columns[col_counter1].col_no = col_counter1;
        all_columns[col_counter1].dependency_count = 0;
        all_columns[col_counter1].tier_level = -1;
        for (col_counter2 = 0; col_counter2 < col_counter1; col_counter2++) {
            if (fill_in_temp[col_counter1][col_counter2] != 0) {
                all_columns[col_counter1].dependency_col[all_columns[col_counter1].dependency_count] = col_counter2;
                all_columns[col_counter1].dependency_count++;
            }
        }
    }

    j = 0;
    for (i = 0; i < MATRIX_SIZE; i++) {
        if (all_columns[i].dependency_count == 0) {
            all_columns[i].tier_level = 0;
            cols_left--;
            tiers[0][j] = i;
            j++;
            zero_tier_size++;
        }
    }

    current_tier = 1;
    int threshold = (int) MATRIX_SIZE * 0.002;
    while (current_tier < threshold) {
        current_tier_size = 0;
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (all_columns[i].tier_level == -1 && check_sat(all_columns[i], current_tier)) {
                all_columns[i].tier_level = current_tier;
                tiers[current_tier][current_tier_size] = i;
                current_tier_size++;
            }
        }
        current_tier++;
    }

    for (i = 0; i < MATRIX_SIZE; i++) {
        if (all_columns[i].tier_level == -1) {
            all_columns[i].tier_level = current_tier;
            tiers[current_tier][current_tier_size] = 1;
            current_tier++;
        }
    }

    memcpy(all_columns_orig, all_columns, sizeof(struct node_info) * MATRIX_SIZE);
    //sort by tier info
    int low = 0, high = MATRIX_SIZE - 1;
    for (i = 0; i < MATRIX_SIZE; i++) {
        all_nodes_sortmap[i].tier_level_origin = all_columns[i].tier_level;
        all_nodes_sortmap[i].col_no = all_columns[i].col_no;
    }
    quick_sort(all_nodes_sortmap, low, high);

    struct node_info *temp;
    temp = (struct node_info *) malloc(sizeof(struct node_info) * MATRIX_SIZE);
    int id;
    for (i = 0; i < MATRIX_SIZE; i++) {
        id = all_nodes_sortmap[i].col_no;
        memcpy(&temp[i], &all_columns[id], sizeof(struct node_info));
    }
    all_columns = temp;

}

void quick_sort(struct tier_map *all_nodes_sortmap, int low, int high) {
    int pi;
    if (low < high) {
        pi = partition(all_nodes_sortmap, low, high);

        quick_sort(all_nodes_sortmap, low, pi - 1);
        quick_sort(all_nodes_sortmap, pi + 1, high);
    }
}

int partition(struct tier_map *all_nodes_sortmap, int low, int high) {
    int i, j;
    struct tier_map temp;
    int target;

    target = all_nodes_sortmap[high].tier_level_origin;
    i = low;

    for (j = low; j < high; j++) {
        if (all_nodes_sortmap[j].tier_level_origin <= target) {
            //swap arr[i] and arr[j]
            temp.tier_level_origin = all_nodes_sortmap[i].tier_level_origin;
            temp.col_no = all_nodes_sortmap[i].col_no;

            all_nodes_sortmap[i].tier_level_origin = all_nodes_sortmap[j].tier_level_origin;
            all_nodes_sortmap[i].col_no = all_nodes_sortmap[j].col_no;

            all_nodes_sortmap[j].tier_level_origin = temp.tier_level_origin;
            all_nodes_sortmap[j].col_no = temp.col_no;

            i++; //increase the smaller number
        }
    }

    //swap arr[i] and arr[high]
    temp.tier_level_origin = all_nodes_sortmap[i].tier_level_origin;
    temp.col_no = all_nodes_sortmap[i].col_no;

    all_nodes_sortmap[i].tier_level_origin = all_nodes_sortmap[j].tier_level_origin;
    all_nodes_sortmap[i].col_no = all_nodes_sortmap[j].col_no;

    all_nodes_sortmap[j].tier_level_origin = temp.tier_level_origin;
    all_nodes_sortmap[j].col_no = temp.col_no;

    return i;

}

