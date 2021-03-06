//
// Created by lshadown on 18.01.2020.
//

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <omp.h>
/* Include polybench common header. */
#include "../polybench.h"

/* Include benchmark-specific header. */
#include "gemver.h"

#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/* Array initialization. */
static
void init_array (int n,
                 DATA_TYPE *alpha,
                 DATA_TYPE *beta,
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
        DATA_TYPE POLYBENCH_1D(u1,N,n),
DATA_TYPE POLYBENCH_1D(v1,N,n),
        DATA_TYPE POLYBENCH_1D(u2,N,n),
DATA_TYPE POLYBENCH_1D(v2,N,n),
        DATA_TYPE POLYBENCH_1D(w,N,n),
DATA_TYPE POLYBENCH_1D(x,N,n),
        DATA_TYPE POLYBENCH_1D(y,N,n),
DATA_TYPE POLYBENCH_1D(z,N,n))
{
int i, j;

*alpha = 1.5;
*beta = 1.2;

DATA_TYPE fn = (DATA_TYPE)n;

for (i = 0; i < n; i++)
{
u1[i] = i;
u2[i] = ((i+1)/fn)/2.0;
v1[i] = ((i+1)/fn)/4.0;
v2[i] = ((i+1)/fn)/6.0;
y[i] = ((i+1)/fn)/8.0;
z[i] = ((i+1)/fn)/9.0;
x[i] = 0.0;
w[i] = 0.0;
for (j = 0; j < n; j++)
A[i][j] = (DATA_TYPE) (i*j % n) / n;
}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_1D(w,N,n))
{
int i;

POLYBENCH_DUMP_START;
POLYBENCH_DUMP_BEGIN("w");
for (i = 0; i < n; i++) {
if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
}
POLYBENCH_DUMP_END("w");
POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
                   DATA_TYPE alpha,
                   DATA_TYPE beta,
                   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
        DATA_TYPE POLYBENCH_1D(u1,N,n),
DATA_TYPE POLYBENCH_1D(v1,N,n),
        DATA_TYPE POLYBENCH_1D(u2,N,n),
DATA_TYPE POLYBENCH_1D(v2,N,n),
        DATA_TYPE POLYBENCH_1D(w,N,n),
DATA_TYPE POLYBENCH_1D(x,N,n),
        DATA_TYPE POLYBENCH_1D(y,N,n),
DATA_TYPE POLYBENCH_1D(z,N,n))
{
int i, j;

#pragma scop

for (i = 0; i < _PB_N; i++)
for (j = 0; j < _PB_N; j++)
A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

for (i = 0; i < _PB_N; i++)
for (j = 0; j < _PB_N; j++)
x[i] = x[i] + beta * A[j][i] * y[j];

for (i = 0; i < _PB_N; i++)
x[i] = x[i] + z[i];

for (i = 0; i < _PB_N; i++)
for (j = 0; j < _PB_N; j++)
w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}

static
void kernel_gemver_trans_pa(int n,
                   DATA_TYPE POLYBENCH_2D(AT,N,N,n,n),
                   DATA_TYPE alpha,
                   DATA_TYPE beta,
                   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                   DATA_TYPE POLYBENCH_1D(u1,N,n),
                   DATA_TYPE POLYBENCH_1D(v1,N,n),
                   DATA_TYPE POLYBENCH_1D(u2,N,n),
                   DATA_TYPE POLYBENCH_1D(v2,N,n),
                   DATA_TYPE POLYBENCH_1D(w,N,n),
                   DATA_TYPE POLYBENCH_1D(x,N,n),
                   DATA_TYPE POLYBENCH_1D(y,N,n),
                   DATA_TYPE POLYBENCH_1D(z,N,n))
{
    int i, j;

#pragma omp parallel for private(i,j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

#pragma omp parallel for private(i,j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            x[i] = x[i] + beta * AT[i][j] * y[j];
        }
    }

#pragma omp parallel for private(i)
    for (i = 0; i < _PB_N; i++) {
        x[i] = x[i] + z[i];
    }

#pragma omp parallel for private(i,j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            w[i] = w[i] + alpha * A[i][j] * x[j];
        }
    }
}

static
void kernel_gemver_tile(int n,
                            DATA_TYPE alpha,
                            DATA_TYPE beta,
                            DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                            DATA_TYPE POLYBENCH_1D(u1,N,n),
                            DATA_TYPE POLYBENCH_1D(v1,N,n),
                            DATA_TYPE POLYBENCH_1D(u2,N,n),
                            DATA_TYPE POLYBENCH_1D(v2,N,n),
                            DATA_TYPE POLYBENCH_1D(w,N,n),
                            DATA_TYPE POLYBENCH_1D(x,N,n),
                            DATA_TYPE POLYBENCH_1D(y,N,n),
                            DATA_TYPE POLYBENCH_1D(z,N,n))
{
    //pluto
    int t1, t2, t3, t4, t5, t6;
    register int lbv, ubv;
/* Start of CLooG code */
    if (_PB_N >= 1) {
        for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
            for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
                for (t4=32*t3;t4<=min(_PB_N-1,32*t3+31);t4++) {
                    lbv=32*t2;
                    ubv=min(_PB_N-1,32*t2+31);
#pragma ivdep
#pragma vector always
                    for (t5=lbv;t5<=ubv;t5++) {
                        A[t4][t5] = A[t4][t5] + u1[t4] * v1[t5] + u2[t4] * v2[t5];;
                        x[t5] = x[t5] + beta * A[t4][t5] * y[t4];;
                    }
                }
            }
        }
        for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
            lbv=32*t2;
            ubv=min(_PB_N-1,32*t2+31);
#pragma ivdep
#pragma vector always
            for (t3=lbv;t3<=ubv;t3++) {
                x[t3] = x[t3] + z[t3];;
            }
        }
        for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
            for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
                for (t4=32*t2;t4<=min(_PB_N-1,32*t2+31);t4++) {
                    for (t5=32*t3;t5<=min(_PB_N-1,32*t3+31);t5++) {
                        w[t4] = w[t4] + alpha * A[t4][t5] * x[t5];;
                    }
                }
            }
        }
    }

}

static
void kernel_gemver_tile_trans(int n,
                        DATA_TYPE POLYBENCH_2D(AT,N,N,n,n),
                        DATA_TYPE alpha,
                        DATA_TYPE beta,
                        DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                        DATA_TYPE POLYBENCH_1D(u1,N,n),
                        DATA_TYPE POLYBENCH_1D(v1,N,n),
                        DATA_TYPE POLYBENCH_1D(u2,N,n),
                        DATA_TYPE POLYBENCH_1D(v2,N,n),
                        DATA_TYPE POLYBENCH_1D(w,N,n),
                        DATA_TYPE POLYBENCH_1D(x,N,n),
                        DATA_TYPE POLYBENCH_1D(y,N,n),
                        DATA_TYPE POLYBENCH_1D(z,N,n))
{
    //pluto
    int t1, t2, t3, t4, t5, t6;
    register int lbv, ubv;
/* Start of CLooG code */
    if (_PB_N >= 1) {
        for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
            for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
                for (t4=32*t2;t4<=min(_PB_N-1,32*t2+31);t4++) {
                    for (t5=32*t3;t5<=min(_PB_N-1,32*t3+31);t5++) {
                        x[t4] = x[t4] + beta * AT[t4][t5] * y[t5];;
                    }
                }
            }
        }
        for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
            lbv=32*t2;
            ubv=min(_PB_N-1,32*t2+31);
#pragma ivdep
#pragma vector always
            for (t3=lbv;t3<=ubv;t3++) {
                x[t3] = x[t3] + z[t3];;
            }
        }
        for (t2=0;t2<=floord(_PB_N-1,32);t2++) {
            for (t3=0;t3<=floord(_PB_N-1,32);t3++) {
                for (t4=32*t2;t4<=min(_PB_N-1,32*t2+31);t4++) {
                    for (t5=32*t3;t5<=min(_PB_N-1,32*t3+31);t5++) {
                        A[t4][t5] = A[t4][t5] + u1[t4] * v1[t5] + u2[t4] * v2[t5];;
                        w[t4] = w[t4] + alpha * A[t4][t5] * x[t5];;
                    }
                }
            }
        }
    }

}

static
void kernel_gemver_pa(int n,
                   DATA_TYPE alpha,
                   DATA_TYPE beta,
                   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                   DATA_TYPE POLYBENCH_1D(u1,N,n),
                   DATA_TYPE POLYBENCH_1D(v1,N,n),
                   DATA_TYPE POLYBENCH_1D(u2,N,n),
                   DATA_TYPE POLYBENCH_1D(v2,N,n),
                   DATA_TYPE POLYBENCH_1D(w,N,n),
                   DATA_TYPE POLYBENCH_1D(x,N,n),
                   DATA_TYPE POLYBENCH_1D(y,N,n),
                   DATA_TYPE POLYBENCH_1D(z,N,n))
{
    int i, j;

#pragma omp parallel for private(i,j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

#pragma omp parallel for private(i,j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            x[i] = x[i] + beta * A[j][i] * y[j];
        }
    }

#pragma omp parallel for private(i)
    for (i = 0; i < _PB_N; i++) {
        x[i] = x[i] + z[i];
    }

#pragma omp parallel for private(i,j)
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            w[i] = w[i] + alpha * A[i][j] * x[j];
        }
    }
}


double run_gemever()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


    /* Initialize array(s). */
    init_array (n, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1),
                POLYBENCH_ARRAY(u2),
                POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_gemver (n, alpha, beta,
                   POLYBENCH_ARRAY(A),
                   POLYBENCH_ARRAY(u1),
                   POLYBENCH_ARRAY(v1),
                   POLYBENCH_ARRAY(u2),
                   POLYBENCH_ARRAY(v2),
                   POLYBENCH_ARRAY(w),
                   POLYBENCH_ARRAY(x),
                   POLYBENCH_ARRAY(y),
                   POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v2);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);

    return result;
}

double run_gemever_pa()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


    /* Initialize array(s). */
    init_array (n, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1),
                POLYBENCH_ARRAY(u2),
                POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_gemver_pa (n, alpha, beta,
                   POLYBENCH_ARRAY(A),
                   POLYBENCH_ARRAY(u1),
                   POLYBENCH_ARRAY(v1),
                   POLYBENCH_ARRAY(u2),
                   POLYBENCH_ARRAY(v2),
                   POLYBENCH_ARRAY(w),
                   POLYBENCH_ARRAY(x),
                   POLYBENCH_ARRAY(y),
                   POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v2);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);

    return result;
}

double run_gemever_tile()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


    /* Initialize array(s). */
    init_array (n, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1),
                POLYBENCH_ARRAY(u2),
                POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_gemver_tile (n, alpha, beta,
                      POLYBENCH_ARRAY(A),
                      POLYBENCH_ARRAY(u1),
                      POLYBENCH_ARRAY(v1),
                      POLYBENCH_ARRAY(u2),
                      POLYBENCH_ARRAY(v2),
                      POLYBENCH_ARRAY(w),
                      POLYBENCH_ARRAY(x),
                      POLYBENCH_ARRAY(y),
                      POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v2);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);

    return result;
}

double run_gemever_tile_trans()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(AT, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


    /* Initialize array(s). */
    init_array (n, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1),
                POLYBENCH_ARRAY(u2),
                POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));


    //double AT[n][n];


    for(int i=0; i< _PB_N; i++){
        for(int j=0; j < _PB_N; j++){
            (*AT)[j][i] = (*A)[i][j];
        }
    }

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_gemver_tile_trans (n, POLYBENCH_ARRAY(AT), alpha, beta,
                            POLYBENCH_ARRAY(A),
                            POLYBENCH_ARRAY(u1),
                            POLYBENCH_ARRAY(v1),
                            POLYBENCH_ARRAY(u2),
                            POLYBENCH_ARRAY(v2),
                            POLYBENCH_ARRAY(w),
                            POLYBENCH_ARRAY(x),
                            POLYBENCH_ARRAY(y),
                            POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(AT);

    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v2);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);

    return result;
}

double run_gemever_trans_pa()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(AT, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


    /* Initialize array(s). */
    init_array (n, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1),
                POLYBENCH_ARRAY(u2),
                POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));


    //double AT[n][n];


    for(int i=0; i< _PB_N; i++){
        for(int j=0; j < _PB_N; j++){
            (*AT)[j][i] = (*A)[i][j];
        }
    }

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_gemver_trans_pa (n, POLYBENCH_ARRAY(AT), alpha, beta,
                      POLYBENCH_ARRAY(A),
                      POLYBENCH_ARRAY(u1),
                      POLYBENCH_ARRAY(v1),
                      POLYBENCH_ARRAY(u2),
                      POLYBENCH_ARRAY(v2),
                      POLYBENCH_ARRAY(w),
                      POLYBENCH_ARRAY(x),
                      POLYBENCH_ARRAY(y),
                      POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(AT);

    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v2);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);

    return result;
}

