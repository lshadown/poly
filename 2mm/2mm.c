//
// Created by lshadown on 19.01.2020.
//

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "../polybench.h"

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
                DATA_TYPE *alpha,
                DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
        DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
int i, j;

*alpha = 1.5;
*beta = 1.2;
for (i = 0; i < ni; i++)
for (j = 0; j < nk; j++)
A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
for (i = 0; i < nk; i++)
for (j = 0; j < nj; j++)
B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
for (i = 0; i < nj; i++)
for (j = 0; j < nl; j++)
C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
for (i = 0; i < ni; i++)
for (j = 0; j < nl; j++)
D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
                 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
int i, j;

POLYBENCH_DUMP_START;
POLYBENCH_DUMP_BEGIN("D");
for (i = 0; i < ni; i++)
for (j = 0; j < nl; j++) {
if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
}
POLYBENCH_DUMP_END("D");
POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
                DATA_TYPE alpha,
                DATA_TYPE beta,
                DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
int i, j, k;

#pragma scop
/* D := alpha*A*B*C + beta*D */
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NJ; j++)
{
tmp[i][j] = SCALAR_VAL(0.0);
for (k = 0; k < _PB_NK; ++k)
tmp[i][j] += alpha * A[i][k] * B[k][j];
}
for (i = 0; i < _PB_NI; i++)
for (j = 0; j < _PB_NL; j++)
{
D[i][j] *= beta;
for (k = 0; k < _PB_NJ; ++k)
D[i][j] += tmp[i][k] * C[k][j];
}
#pragma endscop

}

static
void kernel_2mm_trans_pa(int ni, int nj, int nk, int nl,
                DATA_TYPE alpha,
                DATA_TYPE beta,
                DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                DATA_TYPE POLYBENCH_2D(BT,NK,NJ,nk,nj),
                DATA_TYPE POLYBENCH_2D(CT,NJ,NL,nj,nl),
                DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j, k;
#pragma omp parallel for private(i,k,j)
    for (i = 0; i < _PB_NI; i++){
        for (j = 0; j < _PB_NJ; j++){
            tmp[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_NK; ++k){
                tmp[i][j] = tmp[i][j] + alpha * A[i][k] * BT[j][k];
            }
        }
    }
#pragma omp parallel for private(i,k,j)
    for (i = 0; i < _PB_NI; i++){
        for (j = 0; j < _PB_NL; j++){
            D[i][j] = D[i][j] * beta;
            for (k = 0; k < _PB_NJ; ++k){
                D[i][j] = D[i][j] + tmp[i][k] * CT[j][k];
            }
        }
    }

}


static
void kernel_2mm_pa(int ni, int nj, int nk, int nl,
                DATA_TYPE alpha,
                DATA_TYPE beta,
                DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int i, j, k;



#pragma omp parallel for private(i,k,j)
    for (i = 0; i < _PB_NI; i++){
        for (j = 0; j < _PB_NJ; j++){
            tmp[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < _PB_NK; ++k){
                tmp[i][j] = tmp[i][j] + alpha * A[i][k] * B[k][j];
            }
        }
    }
#pragma omp parallel for private(i,k,j)
    for (i = 0; i < _PB_NI; i++){
        for (j = 0; j < _PB_NL; j++){
            D[i][j] = D[i][j] * beta;
            for (k = 0; k < _PB_NJ; ++k){
                D[i][j] = D[i][j] + tmp[i][k] * C[k][j];
            }
        }
    }

}


double run_2mm()
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
    POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

    /* Initialize array(s). */
    init_array (ni, nj, nk, nl, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B),
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(D));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_2mm (ni, nj, nk, nl,
                alpha, beta,
                POLYBENCH_ARRAY(tmp),
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B),
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(D));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);

    return result;
}

double run_2mm_pa()
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
    POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

    /* Initialize array(s). */
    init_array (ni, nj, nk, nl, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B),
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(D));

    /* Start timer. */
    polybench_timer_start();

    /* Run kernel. */
    kernel_2mm_pa (ni, nj, nk, nl,
                alpha, beta,
                POLYBENCH_ARRAY(tmp),
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B),
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(D));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);

    return result;
}

double run_2mm_trans_pa()
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(BT,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
    POLYBENCH_2D_ARRAY_DECL(CT,DATA_TYPE,NJ,NL,nj,nl);
    POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

    /* Initialize array(s). */
    init_array (ni, nj, nk, nl, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B),
                POLYBENCH_ARRAY(C),
                POLYBENCH_ARRAY(D));

    /* Start timer. */
    polybench_timer_start();

    for (int j = 0; j < _PB_NJ; j++){
        for (int k = 0; k < _PB_NK; ++k){
            (*BT)[k][j] = (*B)[j][k];
        }
    }

    for (int j = 0; j < _PB_NL; j++){
        for (int k = 0; k < _PB_NJ; k++){
            //(*CT)[k][j] = (*C)[j][k];
        }
    }

    /* Run kernel. */
    kernel_2mm_trans_pa (ni, nj, nk, nl,
                   alpha, beta,
                   POLYBENCH_ARRAY(tmp),
                   POLYBENCH_ARRAY(A),
                   POLYBENCH_ARRAY(B),
                   POLYBENCH_ARRAY(C),
                   POLYBENCH_ARRAY(BT),
                   POLYBENCH_ARRAY(CT),
                   POLYBENCH_ARRAY(D));

    /* Stop and print timer. */
    polybench_timer_stop();
    polybench_timer_print();
    double result = polybench_get_timer();

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    //polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(CT);
    POLYBENCH_FREE_ARRAY(BT);

    return result;
}
