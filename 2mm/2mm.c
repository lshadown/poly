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

#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


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
void kernel_2mm_tile(int ni, int nj, int nk, int nl,
                         DATA_TYPE alpha,
                         DATA_TYPE beta,
                         DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                         DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                         DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                         DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                         DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    int t1, t2, t3, t4, t5, t6, t7, t8, t9;
    int lb, ub, lbp, ubp, lb2, ub2;
    register int lbv, ubv;
/* Start of CLooG code */
    if (_PB_NI >= 1) {
        lbp=0;
        ubp=floord(_PB_NI-1,32);
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
        for (t2=lbp;t2<=ubp;t2++) {
            if ((_PB_NJ >= 0) && (_PB_NL >= 0)) {
                for (t3=0;t3<=floord(_PB_NJ+_PB_NL-1,32);t3++) {
                    if ((_PB_NJ >= _PB_NL+1) && (t3 <= floord(_PB_NL-1,32)) && (t3 >= ceild(_PB_NL-31,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=_PB_NL-1;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                            lbv=_PB_NL;
                            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((_PB_NJ >= _PB_NL+1) && (t3 <= floord(_PB_NL-32,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=32*t3+31;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((_PB_NJ <= _PB_NL-1) && (t3 <= floord(_PB_NJ-1,32)) && (t3 >= ceild(_PB_NJ-31,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=_PB_NJ-1;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                            lbv=_PB_NJ;
                            ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                            }
                        }
                    }
                    if ((_PB_NJ <= _PB_NL-1) && (t3 <= floord(_PB_NJ-32,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=32*t3+31;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((_PB_NJ == _PB_NL) && (t3 <= floord(_PB_NJ-1,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((t3 <= floord(_PB_NJ-1,32)) && (t3 >= ceild(_PB_NL,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((t3 <= floord(_PB_NL-1,32)) && (t3 >= ceild(_PB_NJ,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                            }
                        }
                    }
                }
            }
            if (_PB_NL <= -1) {
                for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
                    for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                        lbv=32*t3;
                        ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t5=lbv;t5<=ubv;t5++) {
                            tmp[t4][t5] = SCALAR_VAL(0.0);;
                        }
                    }
                }
            }
            if (_PB_NJ <= -1) {
                for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
                    for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                        lbv=32*t3;
                        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t5=lbv;t5<=ubv;t5++) {
                            D[t4][t5] = D[t4][t5] * beta;;
                        }
                    }
                }
            }
        }
        if (_PB_NJ >= 1) {
            lbp=0;
            ubp=floord(_PB_NI-1,32);
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
                    if (_PB_NK >= 1) {
                        for (t5=0;t5<=floord(_PB_NK-1,32);t5++) {
                            for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
                                for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                                    for (t9=32*t5;t9<=min(_PB_NK-1,32*t5+31);t9++) {
                                        tmp[t6][t7] = tmp[t6][t7] + alpha * A[t6][t9] * B[t9][t7];;
                                    }
                                }
                            }
                        }
                    }
                    if (_PB_NL >= 1) {
                        for (t5=0;t5<=floord(_PB_NL-1,32);t5++) {
                            for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
                                for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                                    lbv=32*t5;
                                    ubv=min(_PB_NL-1,32*t5+31);
#pragma ivdep
#pragma vector always
                                    for (t9=lbv;t9<=ubv;t9++) {
                                        D[t6][t9] = D[t6][t9] + tmp[t6][t7] * C[t7][t9];;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static
void kernel_2mm_tile_trans(int ni, int nj, int nk, int nl,
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
    int t1, t2, t3, t4, t5, t6, t7, t8, t9;
    int lb, ub, lbp, ubp, lb2, ub2;
    register int lbv, ubv;
/* Start of CLooG code */
    if (_PB_NI >= 1) {
        lbp=0;
        ubp=floord(_PB_NI-1,32);
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
        for (t2=lbp;t2<=ubp;t2++) {
            if ((_PB_NJ >= 0) && (_PB_NL >= 0)) {
                for (t3=0;t3<=floord(_PB_NJ+_PB_NL-1,32);t3++) {
                    if ((_PB_NJ >= _PB_NL+1) && (t3 <= floord(_PB_NL-1,32)) && (t3 >= ceild(_PB_NL-31,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=_PB_NL-1;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                            lbv=_PB_NL;
                            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((_PB_NJ >= _PB_NL+1) && (t3 <= floord(_PB_NL-32,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=32*t3+31;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((_PB_NJ <= _PB_NL-1) && (t3 <= floord(_PB_NJ-1,32)) && (t3 >= ceild(_PB_NJ-31,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=_PB_NJ-1;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                            lbv=_PB_NJ;
                            ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                            }
                        }
                    }
                    if ((_PB_NJ <= _PB_NL-1) && (t3 <= floord(_PB_NJ-32,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=32*t3+31;
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((_PB_NJ == _PB_NL) && (t3 <= floord(_PB_NJ-1,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((t3 <= floord(_PB_NJ-1,32)) && (t3 >= ceild(_PB_NL,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                tmp[t4][t5] = SCALAR_VAL(0.0);;
                            }
                        }
                    }
                    if ((t3 <= floord(_PB_NL-1,32)) && (t3 >= ceild(_PB_NJ,32))) {
                        for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                            lbv=32*t3;
                            ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
                            for (t5=lbv;t5<=ubv;t5++) {
                                D[t4][t5] = D[t4][t5] * beta;;
                            }
                        }
                    }
                }
            }
            if (_PB_NL <= -1) {
                for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
                    for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                        lbv=32*t3;
                        ubv=min(_PB_NJ-1,32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t5=lbv;t5<=ubv;t5++) {
                            tmp[t4][t5] = SCALAR_VAL(0.0);;
                        }
                    }
                }
            }
            if (_PB_NJ <= -1) {
                for (t3=0;t3<=floord(_PB_NL-1,32);t3++) {
                    for (t4=32*t2;t4<=min(_PB_NI-1,32*t2+31);t4++) {
                        lbv=32*t3;
                        ubv=min(_PB_NL-1,32*t3+31);
#pragma ivdep
#pragma vector always
                        for (t5=lbv;t5<=ubv;t5++) {
                            D[t4][t5] = D[t4][t5] * beta;;
                        }
                    }
                }
            }
        }
        if (_PB_NJ >= 1) {
            lbp=0;
            ubp=floord(_PB_NI-1,32);
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
            for (t2=lbp;t2<=ubp;t2++) {
                for (t3=0;t3<=floord(_PB_NJ-1,32);t3++) {
                    if (_PB_NK >= 1) {
                        for (t5=0;t5<=floord(_PB_NK-1,32);t5++) {
                            for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
                                for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                                    for (t9=32*t5;t9<=min(_PB_NK-1,32*t5+31);t9++) {
                                        tmp[t6][t7] = tmp[t6][t7] + alpha * A[t6][t9] * BT[t7][t9];;
                                    }
                                }
                            }
                        }
                    }
                    if (_PB_NL >= 1) {
                        for (t5=0;t5<=floord(_PB_NL-1,32);t5++) {
                            for (t6=32*t2;t6<=min(_PB_NI-1,32*t2+31);t6++) {
                                for (t7=32*t3;t7<=min(_PB_NJ-1,32*t3+31);t7++) {
                                    lbv=32*t5;
                                    ubv=min(_PB_NL-1,32*t5+31);
#pragma ivdep
#pragma vector always
                                    for (t9=lbv;t9<=ubv;t9++) {
                                        D[t6][t9] = D[t6][t9] + tmp[t6][t7] * CT[t9][t7];;
                                    }
                                }
                            }
                        }
                    }
                }
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

double run_2mm_tile()
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
    kernel_2mm_tile (ni, nj, nk, nl,
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

double run_2mm_tile_trans()
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
    kernel_2mm_tile_trans (ni, nj, nk, nl,
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
