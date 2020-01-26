//
// Created by lshadown on 18.01.2020.
//
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "../polybench.h"

/* Include benchmark-specific header. */
#include "cholesky.h"

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/* Array initialization. */
static
void init_array(int n,
                DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j;

for (i = 0; i < n; i++)
{
for (j = 0; j <= i; j++)
A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
for (j = i+1; j < n; j++) {
A[i][j] = 0;
}
A[i][i] = 1;
}

/* Make the matrix positive semi-definite. */
int r,s,t;
POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
for (r = 0; r < n; ++r)
for (s = 0; s < n; ++s)
(POLYBENCH_ARRAY(B))[r][s] = 0;
for (t = 0; t < n; ++t)
for (r = 0; r < n; ++r)
for (s = 0; s < n; ++s)
(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
for (r = 0; r < n; ++r)
for (s = 0; s < n; ++s)
A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
int i, j;

POLYBENCH_DUMP_START;
POLYBENCH_DUMP_BEGIN("A");
for (i = 0; i < n; i++)
for (j = 0; j <= i; j++) {
if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
}
POLYBENCH_DUMP_END("A");
POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky(int n,
                     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
int i, j, k;


#pragma scop
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < i; j++) {
            for (k = 0; k < j; k++) {
                A[i][j] -= A[i][k] * A[j][k];
            }
            A[i][j] /= A[j][j];
        }
        for (k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k];
        }
        A[i][i] = SQRT_FUN(A[i][i]);
    }
#pragma endscop

}

static
void kernel_cholesky_pa(int n,
                     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j, k;


    int t1, t2, t3;
    int lb, ub, lbp, ubp, lb2, ub2;
    register int lbv, ubv;
/* Start of CLooG code */
    if (_PB_N >= 1) {
        A[0][0] = SQRT_FUN(A[0][0]);;
        for (t1=1;t1<=min(2,_PB_N-1);t1++) {
            A[t1][0] = A[t1][0] / A[0][0];;
            A[t1][t1] = A[t1][t1] - A[t1][0] * A[t1][0];;
        }
        if (_PB_N >= 3) {
            A[1][1] = SQRT_FUN(A[1][1]);;
        }
        for (t1=3;t1<=_PB_N-1;t1++) {
            A[t1][0] = A[t1][0] / A[0][0];;
            A[t1][t1] = A[t1][t1] - A[t1][0] * A[t1][0];;
            lbp=1;
            ubp=floord(t1-1,2);
#pragma omp parallel for private(lbv,ubv,t3)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=0;t3<=t2-1;t3++) {
                    A[(t1-t2)][t2]=A[(t1-t2)][t2]-A[(t1-t2)][t3]*A[t2][t3];;
                }
                A[(t1-t2)][t2] = A[(t1-t2)][t2] / A[t2][t2];;
                A[(t1-t2)][(t1-t2)] = A[(t1-t2)][(t1-t2)] - A[(t1-t2)][t2] * A[(t1-t2)][t2];;
            }
            if (t1%2 == 0) {
                A[(t1/2)][(t1/2)] = SQRT_FUN(A[(t1/2)][(t1/2)]);;
            }
        }
        for (t1=_PB_N;t1<=2*_PB_N-3;t1++) {
            lbp=t1-_PB_N+1;
            ubp=floord(t1-1,2);
#pragma omp parallel for private(lbv,ubv,t3)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=0;t3<=t2-1;t3++) {
                    A[(t1-t2)][t2]=A[(t1-t2)][t2]-A[(t1-t2)][t3]*A[t2][t3];;
                }
                A[(t1-t2)][t2] = A[(t1-t2)][t2] / A[t2][t2];;
                A[(t1-t2)][(t1-t2)] = A[(t1-t2)][(t1-t2)] - A[(t1-t2)][t2] * A[(t1-t2)][t2];;
            }
            if (t1%2 == 0) {
                A[(t1/2)][(t1/2)] = SQRT_FUN(A[(t1/2)][(t1/2)]);;
            }
        }
        if (_PB_N >= 2) {
            A[(_PB_N-1)][(_PB_N-1)] = SQRT_FUN(A[(_PB_N-1)][(_PB_N-1)]);;
        }
    }

}

static
void kernel_cholesky_trans_pa(int n,
                        DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j, k;

    int t1, t2, t3;
    int lb, ub, lbp, ubp, lb2, ub2;
    register int lbv, ubv;
/* Start of CLooG code */
    if (_PB_N >= 1) {
        A[0][0] = SQRT_FUN(A[0][0]);;
        for (t1=1;t1<=min(2,_PB_N-1);t1++) {
            A[t1][0] = A[t1][0] / A[0][0];;
            A[t1][t1] = A[t1][t1] - A[t1][0] * A[t1][0];;
        }
        if (_PB_N >= 3) {
            A[1][1] = SQRT_FUN(A[1][1]);;
        }
        for (t1=3;t1<=_PB_N-1;t1++) {
            A[t1][0] = A[t1][0] / A[0][0];;
            A[t1][t1] = A[t1][t1] - A[t1][0] * A[t1][0];;
            lbp=1;
            ubp=floord(t1-1,2);
#pragma omp parallel for private(lbv,ubv,t3)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=0;t3<=t2-1;t3++) {
                    A[(t1-t2)][t2]=A[(t1-t2)][t2]-A[(t1-t2)][t3]*A[t2][t3];;
                }
                A[(t1-t2)][t2] = A[(t1-t2)][t2] / A[t2][t2];;
                A[(t1-t2)][(t1-t2)] = A[(t1-t2)][(t1-t2)] - A[(t1-t2)][t2] * A[(t1-t2)][t2];;
            }
            if (t1%2 == 0) {
                A[(t1/2)][(t1/2)] = SQRT_FUN(A[(t1/2)][(t1/2)]);;
            }
        }
        for (t1=_PB_N;t1<=2*_PB_N-3;t1++) {
            lbp=t1-_PB_N+1;
            ubp=floord(t1-1,2);
#pragma omp parallel for private(lbv,ubv,t3)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=0;t3<=t2-1;t3++) {
                    A[(t1-t2)][t2]=A[(t1-t2)][t2]-A[(t1-t2)][t3]*A[t2][t3];;
                }
                A[(t1-t2)][t2] = A[(t1-t2)][t2] / A[t2][t2];;
                A[(t1-t2)][(t1-t2)] = A[(t1-t2)][(t1-t2)] - A[(t1-t2)][t2] * A[(t1-t2)][t2];;
            }
            if (t1%2 == 0) {
                A[(t1/2)][(t1/2)] = SQRT_FUN(A[(t1/2)][(t1/2)]);;
            }
        }
        if (_PB_N >= 2) {
            A[(_PB_N-1)][(_PB_N-1)] = SQRT_FUN(A[(_PB_N-1)][(_PB_N-1)]);;
        }
    }

}



int run()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

    /* Initialize array(s). */
    init_array (n, POLYBENCH_ARRAY(A));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_cholesky (n, POLYBENCH_ARRAY(A));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);

    return 0;
}

