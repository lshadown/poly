//
// Created by lshadown on 18.01.2020.
//
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
                DATA_TYPE *alpha,
                DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
int i, j;

*alpha = 1.5;
*beta = 1.2;
for (i = 0; i < n; i++)
for (j = 0; j < m; j++) {
A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}
for (i = 0; i < n; i++)
for (j = 0; j < n; j++) {
C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
int i, j;

POLYBENCH_DUMP_START;
POLYBENCH_DUMP_BEGIN("C");
for (i = 0; i < n; i++)
for (j = 0; j < n; j++) {
if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
}
POLYBENCH_DUMP_END("C");
POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syr2k(int n, int m,
                  DATA_TYPE alpha,
                  DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
for (i = 0; i < _PB_N; i++) {
for (j = 0; j <= i; j++)
C[i][j] *= beta;
for (k = 0; k < _PB_M; k++)
for (j = 0; j <= i; j++)
{
C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
}
}
#pragma endscop

}

static
void kernel_syr2k_pa(int n, int m,
                  DATA_TYPE alpha,
                  DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma omp parallel for private(i,k,j)
for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++){
        C[i][j] *= beta;
    }
    for (k = 0; k < _PB_M; k++){
        for (j = 0; j <= i; j++){
            C[i][j] = C[i][j] + A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
        }
    }
}

}

static
void kernel_syr2k_trans_pa(int n, int m,
                     DATA_TYPE alpha,
                     DATA_TYPE beta,
                     DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
                     DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                     DATA_TYPE POLYBENCH_2D(B,N,M,n,m),
                     DATA_TYPE POLYBENCH_2D(BT,N,M,n,m)) {
    int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma omp parallel for private(i, k, j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++) {
            C[i][j] *= beta;
        }
        for (k = 0; k < _PB_M; k++) {
            for (j = 0; j <= i; j++) {
                C[i][j] = C[i][j] + A[j][k] * alpha * B[i][k] + BT[k][j] * alpha * A[i][k];
            }
        }
    }
}


double run_syr2k()
{
    /* Retrieve problem size. */
    int n = N;
    int m = M;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

    /* Initialize array(s). */
    init_array (n, m, &alpha, &beta,
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_syr2k (n, m,
                  alpha, beta,
                  POLYBENCH_ARRAY(C),
                  POLYBENCH_ARRAY(A),
                  POLYBENCH_ARRAY(B));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);

    return result;
}

double run_syr2k_pa()
{
    /* Retrieve problem size. */
    int n = N;
    int m = M;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);

    /* Initialize array(s). */
    init_array (n, m, &alpha, &beta,
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_syr2k_pa (n, m,
                  alpha, beta,
                  POLYBENCH_ARRAY(C),
                  POLYBENCH_ARRAY(A),
                  POLYBENCH_ARRAY(B));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);

    return result;
}

double run_syr2k_trans_pa()
{
    /* Retrieve problem size. */
    int n = N;
    int m = M;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);
    POLYBENCH_2D_ARRAY_DECL(BT,DATA_TYPE,N,M,n,m);

    /* Initialize array(s). */
    init_array (n, m, &alpha, &beta,
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B));

    for(int i=0; i< _PB_N; i++){
        for(int j=0; j < _PB_N; j++){
            (*BT)[j][i] = (*B)[i][j];
        }
    }

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_syr2k_trans_pa (n, m,
                     alpha, beta,
                     POLYBENCH_ARRAY(C),
                     POLYBENCH_ARRAY(A),
                     POLYBENCH_ARRAY(B),
                     POLYBENCH_ARRAY(BT));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);

    return result;
}

