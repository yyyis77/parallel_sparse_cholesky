#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <math.h>
#include <string.h>

#define MATRIX_SIZE 30

//TODO: Refactor code so matrix is read from a file.
double mat_in[MATRIX_SIZE][MATRIX_SIZE] = {{49, 0,   0,   0,   42, 0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   98,  0,   0,  0,   0,    0,   0,   0,   0,   0,   0,},
                                           {0,  361, 342, 0,   0,  0,   0,   0,   0,   190, 0,   285, 0,  0,   0,   0,   0,   0,   0,   0,   361, 0,  0,   0,    0,   0,   0,   0,   0,   0,},
                                           {0,  342, 373, 0,   0,  0,   0,   0,   0,   180, 0,   270, 0,  0,   0,   0,   0,   0,   133, 0,   342, 0,  0,   0,    63,  0,   0,   0,   42,  0,},
                                           {0,  0,   0,   81,  0,  0,   0,   0,   0,   0,   0,   135, 0,  0,   108, 0,   36,  0,   0,   0,   0,   0,  0,   153,  0,   0,   0,   0,   0,   0,},
                                           {42, 0,   0,   0,   37, 0,   18,  0,   0,   6,   0,   0,   0,  0,   0,   0,   15,  0,   0,   84,  0,   0,  0,   0,    0,   0,   6,   0,   0,   0,},
                                           {0,  0,   0,   0,   0,  324, 0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   18,  0,   0,   0,   0,  0,   0,    0,   0,   0,   0,   0,   0,},
                                           {0,  0,   0,   0,   18, 0,   333, 0,   0,   108, 57,  39,  0,  0,   0,   39,  270, 0,   0,   36,  0,   0,  0,   0,    42,  0,   108, 0,   0,   0,},
                                           {0,  0,   0,   0,   0,  0,   0,   81,  0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   63,  0,   0,   0,  0,   0,    135, 0,   0,   126, 0,   90,},
                                           {0,  0,   0,   0,   0,  0,   0,   0,   361, 0,   0,   0,   0,  114, 0,   0,   0,   0,   0,   0,   0,   0,  0,   0,    0,   19,  0,   0,   0,   0,},
                                           {0,  190, 180, 0,   6,  0,   108, 0,   0,   161, 40,  150, 0,  0,   0,   30,  90,  0,   0,   0,   190, 0,  0,   85,   0,   0,   36,  0,   0,   0,},
                                           {0,  0,   0,   0,   0,  0,   57,  0,   0,   40,  569, 247, 0,  0,   0,   295, 0,   0,   0,   228, 0,   24, 0,   136,  266, 0,   0,   0,   96,  0,},
                                           {0,  285, 270, 135, 0,  0,   39,  0,   0,   150, 247, 875, 0,  0,   180, 169, 60,  0,   0,   156, 285, 0,  0,   271,  182, 0,   0,   0,   0,   80,},
                                           {0,  0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   16, 0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,    0,   0,   20,  0,   0,   0,},
                                           {0,  0,   0,   0,   0,  0,   0,   0,   114, 0,   0,   0,   0,  232, 0,   0,   0,   0,   0,   0,   0,   0,  28,  0,    0,   146, 0,   238, 0,   84,},
                                           {0,  0,   0,   108, 0,  0,   0,   0,   0,   0,   0,   180, 0,  0,   244, 0,   48,  0,   0,   0,   0,   0,  200, 204,  40,  0,   10,  0,   0,   0,},
                                           {0,  0,   0,   0,   0,  0,   39,  0,   0,   30,  295, 169, 0,  0,   0,   286, 0,   0,   0,   156, 0,   0,  0,   102,  182, 0,   0,   0,   0,   0,},
                                           {0,  0,   0,   36,  15, 0,   270, 0,   0,   90,  0,   60,  0,  0,   48,  0,   257, 0,   0,   0,   0,   0,  0,   88,   0,   0,   90,  8,   0,   20,},
                                           {0,  0,   0,   0,   0,  18,  0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   401, 0,   0,   0,   60, 100, 0,    0,   0,   0,   0,   0,   0,},
                                           {0,  0,   133, 0,   0,  0,   0,   63,  0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   635, 0,   0,   0,  0,   255,  276, 0,   0,   113, 114, 70,},
                                           {98, 0,   0,   0,   84, 0,   36,  0,   0,   0,   228, 156, 0,  0,   0,   156, 0,   0,   0,   404, 0,   0,  0,   0,    168, 0,   0,   0,   0,   0,},
                                           {0,  361, 342, 0,   0,  0,   0,   0,   0,   190, 0,   285, 0,  0,   0,   0,   0,   0,   0,   0,   410, 0,  0,   0,    0,   0,   56,  0,   0,   0,},
                                           {0,  0,   0,   0,   0,  0,   0,   0,   0,   0,   24,  0,   0,  0,   0,   0,   0,   60,  0,   0,   0,   22, 15,  0,    0,   0,   0,   36,  16,  0,},
                                           {0,  0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,  28,  200, 0,   0,   100, 0,   0,   0,   15, 718, 289,  80,  20,  20,  136, 0,   12,},
                                           {0,  0,   0,   153, 0,  0,   0,   0,   0,   85,  136, 271, 0,  0,   204, 102, 88,  0,   255, 0,   0,   0,  289, 1326, 0,   0,   0,   129, 0,   30,},
                                           {0,  0,   63,  0,   0,  0,   42,  135, 0,   0,   266, 182, 0,  0,   40,  182, 0,   0,   276, 168, 0,   0,  80,  0,    687, 0,   4,   210, 54,  150,},
                                           {0,  0,   0,   0,   0,  0,   0,   0,   19,  0,   0,   0,   0,  146, 0,   0,   0,   0,   0,   0,   0,   0,  20,  0,    0,   462, 0,   170, 0,   60,},
                                           {0,  0,   0,   0,   6,  0,   108, 0,   0,   36,  0,   0,   20, 0,   10,  0,   90,  0,   0,   0,   56,  0,  20,  0,    4,   0,   487, 0,   0,   266,},
                                           {0,  0,   0,   0,   0,  0,   0,   126, 0,   0,   0,   0,   0,  238, 0,   0,   8,   0,   113, 0,   0,   36, 136, 129,  210, 170, 0,   926, 0,   252,},
                                           {0,  0,   42,  0,   0,  0,   0,   0,   0,   0,   96,  0,   0,  0,   0,   0,   0,   0,   114, 0,   0,   16, 0,   0,    54,  0,   0,   0,   356, 0,},
                                           {0,  0,   0,   0,   0,  0,   0,   90,  0,   0,   0,   80,  0,  84,  0,   0,   20,  0,   70,  0,   0,   0,  12,  30,   150, 60,  266, 252, 0,   463,}};


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

int tiers[MATRIX_SIZE][MATRIX_SIZE];
int current_tier;
int current_tier_size;
int zero_tier_size;

struct node_info *all_columns;
struct node_info *all_columns_orig;
struct tier_map *all_columns_sortmap;
int **rank_col_map;
int *iteration_per_rank;
int **col_send_map;
int **col_recv_map;

int has_node_left(struct node_info all_nodes[]);

int check_sat(struct node_info node, int tier);

void dependency_checker(struct tier_map *all_nodes_sorted, int num_proc);

void quick_sort(struct tier_map *all_nodes_sorted, int low, int high);

int partition(struct tier_map *all_nodes_sortmap, int low, int high);

void cdiv(double (*matrix)[MATRIX_SIZE], int col_num_i);

void cmod(double (*matrix)[MATRIX_SIZE], int col_num_j, int col_num_k);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank_id, num_proc;
    int i, j, k, m;

    //rank_id=3;
    //num_proc=4;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Request request;

    zero_tier_size = 0;

    all_columns = malloc(sizeof(struct node_info) * MATRIX_SIZE);
    all_columns_orig = malloc(sizeof(struct node_info) * MATRIX_SIZE);
    struct node_info *current_node;

    all_columns_sortmap = malloc(sizeof(struct tier_map) * MATRIX_SIZE);

    //zero_tier_size is already set here.
    dependency_checker(all_columns_sortmap, num_proc);

    double **buffer_mat = (double **) malloc(sizeof(double *) * MATRIX_SIZE);
    for (i = 0; i < MATRIX_SIZE; i++) {
        buffer_mat[i] = (double *) malloc(sizeof(double *) * MATRIX_SIZE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

/*    if(rank_id==0) {
        for (i = 0; i < MATRIX_SIZE; i++) {
            printf("i:%d col:%d\n", i,all_columns[i].col_no);
        }
    }*/

    float recv[MATRIX_SIZE][MATRIX_SIZE] = {0};
    int available[MATRIX_SIZE] = {0};

    /******************* construct iteration_per_rank *******************/
    int iteration = MATRIX_SIZE / num_proc;
    if (rank_id < MATRIX_SIZE % num_proc) {
        iteration++;
    }
    //printf("%d proc has %d iterations\n",rank_id, iteration);

    if (rank_id == 0) {
        iteration_per_rank = (int *) malloc(sizeof(int) * num_proc);

        iteration_per_rank[0] = iteration;
        for (i = 1; i < num_proc; i++) {
            MPI_Recv(&iteration_per_rank[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }

    } else {
        MPI_Send(&iteration, 1, MPI_INT, 0, rank_id, MPI_COMM_WORLD);
    }

    /******************* construct rank_col_map *******************/
    int temp;
    int *self_cols;
    if (rank_id == 0) {
        rank_col_map = (int **) malloc(sizeof(int *) * num_proc);
        for (temp = 0; temp < num_proc; temp++) {
            rank_col_map[temp] = (int *) malloc(sizeof(int) * iteration_per_rank[temp]);
        }
        self_cols = (int *) malloc(sizeof(int) * iteration);
        j = 0;
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (i % num_proc == 0) {
                self_cols[j] = all_columns[i].col_no;
                j++;
            }
        }

        memcpy(rank_col_map[0], self_cols, sizeof(int) * iteration);
        for (temp = 1; temp < num_proc; temp++) {
            MPI_Irecv(rank_col_map[temp], iteration_per_rank[temp], MPI_INT, temp, temp, MPI_COMM_WORLD, &request);
        }
    } else {
        self_cols = (int *) malloc(sizeof(int) * iteration);
        j = 0;
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (i % num_proc == rank_id) {
                self_cols[j] = all_columns[i].col_no;
                j++;
            }
        }
        MPI_Isend(self_cols, iteration, MPI_INT, 0, rank_id, MPI_COMM_WORLD, &request);
    }
    MPI_Wait(&request, MPI_STATUSES_IGNORE);

    //test allocate cols to processors
/*    if(rank_id==0) {
        for(i=0; i<num_proc;i++) {
            printf("%d rank: ",i);
            for (j = 0; j < iteration_per_rank[i]; j++) {
                printf("%d col  ", rank_col_map[i][j]);
            }
            printf("\n");
        }
    }*/

    /******************* construct col_send/recv_map *******************/
    int iter;
    int recv_rank_id;
    int **need_to_recv = (int **) malloc(sizeof(int *) * iteration);
    for (i = 0; i < iteration; i++) {
        need_to_recv[i] = (int *) malloc(sizeof(int) * num_proc);
        for (j = 0; j < num_proc; j++) {
            need_to_recv[i][j] = 0;
        }
    }

    for (iter = 0; iter < iteration; iter++) {
        current_node = &all_columns_orig[self_cols[iter]];
        for (j = 0; j < current_node->dependency_count; j++) {
            //printf("%d rank current col:%d dependency col:%d\n",rank_id,current_node->col_no,current_node->dependency_col[j]);
            for (k = 0; k < MATRIX_SIZE; k++) {
                if (current_node->dependency_col[j] == all_columns[k].col_no) {
                    recv_rank_id = k % num_proc;
                    //printf("rank %d computing %d send to rank %d\n",rank_id,current_node->col_no, send_rank_id);
                    break;
                }
            }
            if (need_to_recv[iter][recv_rank_id] == 1 || recv_rank_id == rank_id) {
                continue;
            }
            need_to_recv[iter][recv_rank_id] = 1;
            //printf("rank %d computing %d recv from rank %d\n",rank_id,current_node->col_no, recv_rank_id);
        }
    }

    int break_flag = 0;
    struct node_info *target_node;
    int *send_rank_counter = (int *) malloc(sizeof(int) * iteration);
    int **need_to_send = (int **) malloc(sizeof(int *) * iteration);
    for (i = 0; i < iteration; i++) {
        need_to_send[i] = (int *) malloc(sizeof(int) * MATRIX_SIZE);
        for (j = 0; j < MATRIX_SIZE; j++) {
            need_to_send[i][j] = -1;
        }
    }

    for (iter = 0; iter < iteration; iter++) {
        current_node = &all_columns_orig[self_cols[iter]];
        for (j = 0; j < MATRIX_SIZE; j++) {
            target_node = &all_columns_orig[j];
            for (k = 0; k < target_node->dependency_count; k++) {
                if (target_node->dependency_col[k] == current_node->col_no) {
                    if (target_node->col_no % num_proc == rank_id) {
                        break;
                    }
                    //printf("rand %d computing %d sent to rank %d\n",rank_id,current_node->col_no,target_node->col_no%num_proc);
                    for (m = 0; m < send_rank_counter[iter]; m++) {
                        if (need_to_send[iter][m] == target_node->col_no % num_proc) {
                            break_flag = 1;
                            break;
                        }
                    }
                    if (break_flag == 1) {
                        break_flag = 0;
                        break;
                    }
                    need_to_send[iter][send_rank_counter[iter]] = target_node->col_no % num_proc;
                    printf("rand %d computing %d sent to rank %d\n", rank_id, current_node->col_no,
                           target_node->col_no % num_proc);
                    send_rank_counter[iter]++;
                }

            }
        }
    }

    /******************* computation *******************/
    iter;
    int recv_col;
    int send_count;
    int send_rank[MATRIX_SIZE];
    for (iter = 0; iter < iteration; iter++) {
        current_node = &all_columns_orig[self_cols[iter]];
        //printf("rank %d compute %d col at iteration %d\n",rank_id,current_node->col_no,iter);
        if (current_node->tier_level == 0) {
            cdiv(mat_in, current_node->col_no);
            available[current_node->col_no] = 1;

        } else {
            /*******  receive and cmod  ********/
            for (i = 0; i < current_node->dependency_count; i++) {
                recv_col = current_node->dependency_col[i];
                int recv_rank;
                for (j = 0; j < MATRIX_SIZE; j++) {
                    if (all_columns[j].col_no == recv_col) {
                        recv_rank = j % num_proc;
                        break;
                    }
                }
                if (rank_id == recv_rank) {
                    available[current_node->col_no] = 1;
                    cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);
                } else if (available[recv_col] == 0) {
                    //printf("rank %d computing %d want to recv col %d from rank %d\n", rank_id, current_node->col_no, recv_col, recv_rank);
                    MPI_Irecv(buffer_mat[recv_col],
                              MATRIX_SIZE,
                              MPI_DOUBLE,
                              recv_rank,
                              recv_col,
                              MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUSES_IGNORE);
                    available[recv_col] = 1;
                    //printf("rank %d received %d\n",rank_id, recv_col);

                    int ccc;
                    for (ccc = 0; ccc < MATRIX_SIZE; ccc++) {
                        mat_in[ccc][recv_col] = buffer_mat[recv_col][ccc];
                    }

                    cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);
                } else {
                    //printf("rank %d computing %d start without communication\n",rank_id,current_node->col_no);
                    cmod(mat_in, current_node->col_no, current_node->dependency_col[i]);
                }
            }
            cdiv(mat_in, current_node->col_no);
        }

        /*******  send  ********/
        //calculate which rank to send
        k = 0;
        //dependent col must appear after current col
        for (i = current_node->col_no; i < MATRIX_SIZE; i++) {
            for (j = 0; j < all_columns[i].dependency_count; j++) {
                if (all_columns[i].dependency_col[j] == current_node->col_no) {
                    if (rank_id == i % num_proc) {
                        continue;
                    }
                    send_rank[k] = i % num_proc;
                    //printf("rank %d computing %d send to rank %d to compute %d col\n",rank_id,current_node->col_no,send_rank[k],all_columns[i].col_no);
                    k++;
                    break;
                }
            }
        }
        send_count = k;

        for (i = 0; i < MATRIX_SIZE; i++) {
            buffer_mat[current_node->col_no][i] = mat_in[i][current_node->col_no];
        }

        for (i = 0; i < send_count; i++) {
            MPI_Isend(buffer_mat[current_node->col_no],
                      MATRIX_SIZE,
                      MPI_DOUBLE,
                      send_rank[i],
                      current_node->col_no,
                      MPI_COMM_WORLD, &request);
        }
        //printf("self id:%d\n",rank_id);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (iteration * num_proc < MATRIX_SIZE) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // send result to rank 0
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
        for (i = 0; i < iteration; i++) {
            for (j = 0; j < MATRIX_SIZE; j++) {
                buffer_mat[self_cols[i]][j] = mat_in[j][self_cols[i]];
            }
            MPI_Send(buffer_mat[self_cols[i]], MATRIX_SIZE, MPI_DOUBLE, 0, rank_id, MPI_COMM_WORLD);
        }
    }
    MPI_Wait(&request, MPI_STATUSES_IGNORE);

    // test final result
/*    if(rank_id==0)
    {
        for(i=0;i<MATRIX_SIZE;i++)
        {
            for(j=0;j<MATRIX_SIZE;j++)
            {
                printf("%4.0lf",mat_in[i][j]);
            }
            printf("\n");
        }
    }*/

    MPI_Finalize();
}

void cmod(double (*matrix)[MATRIX_SIZE], int col_num_j, int col_num_k) {
    int i, j, k;
    j = col_num_j;
    k = col_num_k;

    for (i = j; i < MATRIX_SIZE; i++) {
        matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[j][k];
    }
}

void cdiv(double (*matrix)[MATRIX_SIZE], int col_num_j) {
    int i, j;
    j = col_num_j;
    matrix[col_num_j][col_num_j] = sqrt(matrix[col_num_j][col_num_j]);

    for (i = j + 1; i < MATRIX_SIZE; i++) {
        matrix[i][j] = matrix[i][j] / matrix[j][j];
    }
}

int has_node_left(struct node_info all_nodes[]) {
    int i;
    int res = 0;
    for (i = 0; i < MATRIX_SIZE; i++) {
        if (all_nodes[i].tier_level == -1)
            return 1;
    }
    return res;
}

int check_sat(struct node_info node, int tier) {
    int i, j, k;
    int dep_count;
    int found_flag = 0;

    for (dep_count = 0; dep_count < node.dependency_count; dep_count++) {
        found_flag = 0;
        for (i = 0; i < tier; i++) {
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

    int i, j, k;
    int fill_in_temp[MATRIX_SIZE][MATRIX_SIZE];

    /*mat_in=(double**) malloc(sizeof(double)*MATRIX_SIZE);
    for(i=0;i<MATRIX_SIZE;i++)
    {
        mat_in[i]=sparse_matrix[i];
    }*/

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

    /*printf("Fill-in Matrix:\n");
    for(i=0;i<MATRIX_SIZE;i++)
    {
        for(j=0;j<MATRIX_SIZE;j++)
        {
            printf("%d ",fill_in_temp[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");*/

    //printf("col 0: \n");
    all_columns[0].col_no = 0;
    all_columns[0].dependency_count = 0;
    all_columns[0].tier_level = -1;

    for (col_counter1 = 1; col_counter1 < MATRIX_SIZE; col_counter1++) {
        //printf("col %d: ",col_counter1);
        all_columns[col_counter1].col_no = col_counter1;
        all_columns[col_counter1].dependency_count = 0;
        all_columns[col_counter1].tier_level = -1;
        for (col_counter2 = 0; col_counter2 < col_counter1; col_counter2++) {
            if (fill_in_temp[col_counter1][col_counter2] != 0) {
                //printf("%d ",col_counter2);
                all_columns[col_counter1].dependency_col[all_columns[col_counter1].dependency_count] = col_counter2;
                all_columns[col_counter1].dependency_count++;
            }
        }
        //printf("\n");
    }

/*
	printf("Dependency Info:\n");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		printf("col %d: ",all_nodes[i].col_no);
		for(j=0;j<all_nodes[i].dependency_count;j++)
		{
			printf("%d ",all_nodes[i].dependency_col[j]);
		}
		printf("\n");
	}
	printf("\n\n");*/

    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            tiers[i][j] == -2;
        }
    }

    j = 0;
    for (i = 0; i < MATRIX_SIZE; i++) {
        if (all_columns[i].dependency_count == 0) {
            all_columns[i].tier_level = 0;
            tiers[0][j] = i;
            j++;
            zero_tier_size++;
        }
    }

    current_tier = 1;
    while (has_node_left(all_columns) == 1) {
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

/*	printf("Tier Info:\n");
	for(i=0;i<MATRIX_SIZE;i++)
	{
		printf("col %d: %d\n",all_nodes[i].col_no,all_nodes[i].tier_level);
	}
	printf("\n\n");*/

    memcpy(all_columns_orig, all_columns, sizeof(struct node_info) * MATRIX_SIZE);
    //sort by tier info
    int low = 0, high = MATRIX_SIZE - 1;
    for (i = 0; i < MATRIX_SIZE; i++) {
        all_nodes_sortmap[i].tier_level_origin = all_columns[i].tier_level;
        all_nodes_sortmap[i].col_no = all_columns[i].col_no;
    }
    quick_sort(all_nodes_sortmap, low, high);

/*    printf("sorted map:\n");
    for(i=0;i<MATRIX_SIZE;i++){
        printf("col %d tier: %d\n",all_nodes_sortmap[i].col_no,all_nodes_sortmap[i].tier_level_origin);
    }
    printf("\n\n");*/

    struct node_info *temp;
    temp = (struct node_info *) malloc(sizeof(struct node_info) * MATRIX_SIZE);
    int id;
    for (i = 0; i < MATRIX_SIZE; i++) {
        id = all_nodes_sortmap[i].col_no;
        memcpy(&temp[i], &all_columns[id], sizeof(struct node_info));
    }
    all_columns = temp;

/*    printf("Tier Info After Sort:\n");
    for(i=0;i<MATRIX_SIZE;i++)
    {
        printf("col %d: %d\n",all_nodes[i].col_no,all_nodes[i].tier_level);
    }
    printf("\n\n");*/



    /*int out_node_count=0;
    int current_tier_count=0;
    current_tier=0;
    while(out_node_count<MATRIX_SIZE)
    {
        printf("Tier %d: ",current_tier);
        current_tier_count=0;
        for(i=0;i<MATRIX_SIZE;i++)
        {
            if(all_nodes[i].tier_level==current_tier) {
                current_tier_count++;
            }
        }
        printf("%d: ",current_tier_count);
        for(i=0;i<MATRIX_SIZE;i++)
        {
            if(all_nodes[i].tier_level==current_tier)
            {
                printf("%d ",i);
                out_node_count++;
            }
        }
        printf("\n");
        current_tier++;
        current_tier_count=0;
    }*/

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