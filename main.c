#include <stdio.h>
#include "cholesky/cholesky.h"
#include "gemver/gemver.h"
#include "symm/symm.h"
#include "syr2k/syr2k.h"
#include "2mm/2mm.h"

double calculateAverage(double *tab, int attempts){
    double sum = 0.0;
    for(int i=0; i < attempts; i++){
        sum = sum + tab[i];
    }
    return (double)sum/attempts;
}

void saveTabToFile(double *tab, int attempts, char *fileName){
    FILE *fp;
    fp = fopen(fileName, "w+");
    if(fp == NULL) //if file does not exist, create it
    {
        fp = fopen(fileName, "wb");
    }
    for(int i=0; i < attempts; i++){
        fprintf(fp, "%f \n", tab[i]);
    }
    fclose(fp);
}

void gemever(int attempt){
    double synchro_tab[attempt];
    double pa_tab[attempt];
    double trans_par_tab[attempt];
    //synchro
    int synchro =0, par =0, trans_par = 0;
    while(synchro != attempt){
        double result = run_gemever();
        synchro_tab[synchro] = result;
        synchro++;
    }
    //pa
    while(par != attempt){
        double result = run_gemever_pa();
        pa_tab[par] = result;
        par++;
    }
    //trans_pa
    while(trans_par != attempt){
        double result = run_gemever_trans_pa();
        trans_par_tab[trans_par] = result;
        trans_par++;
    }

    printf("Synchro execution time: %f \n", calculateAverage(synchro_tab, attempt));
    printf("Parallel  execution time: %f \n", calculateAverage(pa_tab, attempt));
    printf("Transposition + Parallel execution time: %f \n", calculateAverage(trans_par_tab, attempt));

    saveTabToFile(synchro_tab, attempt, "/home/lshadown/Projects/issfResult/gemver_synchro.txt");
    saveTabToFile(pa_tab, attempt,  "/home/lshadown/Projects/issfResult/gemver_pa.txt");
    saveTabToFile(trans_par_tab, attempt, "/home/lshadown/Projects/issfResult/gemver_trans_par.txt");

}

void symm(int attempt){
    double synchro_tab[attempt];
    double pa_tab[attempt];
    double trans_par_tab[attempt];
    //synchro
    int synchro =0, par =0, trans_par = 0;
    while(synchro != attempt){
        double result = run_symm();
        synchro_tab[synchro] = result;
        synchro++;
    }
    //pa
    while(par != attempt){
        double result = run_symm_pa();
        pa_tab[par] = result;
        par++;
    }
    //trans_pa
    /*while(trans_par != attempt){
        double result = run_gemever_trans_pa();
        trans_par_tab[trans_par] = result;
        trans_par++;
    }*/

    printf("Synchro execution time: %f \n", calculateAverage(synchro_tab, attempt));
    printf("Parallel  execution time: %f \n", calculateAverage(pa_tab, attempt));
    //printf("Transposition + Parallel execution time: %f \n", calculateAverage(trans_par_tab, attempt));
    saveTabToFile(synchro_tab, attempt, "/home/lshadown/Projects/issfResult/gemver_synchro.txt");
    saveTabToFile(pa_tab, attempt,  "/home/lshadown/Projects/issfResult/gemver_pa.txt");
    //saveTabToFile(trans_par_tab, attempt, "/home/lshadown/Projects/issfResult/gemver_trans_par.txt");

}

void syr2k(int attempt){
    double synchro_tab[attempt];
    double pa_tab[attempt];
    double trans_par_tab[attempt];
    //synchro
    int synchro =0, par =0, trans_par = 0;
    while(synchro != attempt){
        double result = run_syr2k();
        synchro_tab[synchro] = result;
        synchro++;
    }
    //pa
    while(par != attempt){
        double result = run_syr2k_pa();
        pa_tab[par] = result;
        par++;
    }
    //trans_pa
    while(trans_par != attempt){
        double result = run_gemever_trans_pa();
        trans_par_tab[trans_par] = result;
        trans_par++;
    }

    printf("Synchro execution time: %f \n", calculateAverage(synchro_tab, attempt));
    printf("Parallel  execution time: %f \n", calculateAverage(pa_tab, attempt));
    printf("Transposition + Parallel execution time: %f \n", calculateAverage(trans_par_tab, attempt));
    saveTabToFile(synchro_tab, attempt, "/home/lshadown/Projects/issfResult/syr2k_synchro.txt");
    saveTabToFile(pa_tab, attempt,  "/home/lshadown/Projects/issfResult/syr2k_pa.txt");
    saveTabToFile(trans_par_tab, attempt, "/home/lshadown/Projects/issfResult/syr2k_trans_par.txt");

}

void _2mm(int attempt){
    double synchro_tab[attempt];
    double pa_tab[attempt];
    double trans_par_tab[attempt];
    //synchro
    int synchro =0, par =0, trans_par = 0;
    while(synchro != attempt){
        double result = run_2mm();
        synchro_tab[synchro] = result;
        synchro++;
    }
    //pa
    while(par != attempt){
        double result = run_2mm_pa();
        pa_tab[par] = result;
        par++;
    }
    //trans_pa
    while(trans_par != attempt){
        double result = run_2mm_trans_pa();
        trans_par_tab[trans_par] = result;
        trans_par++;
    }

    printf("Synchro execution time: %f \n", calculateAverage(synchro_tab, attempt));
    printf("Parallel  execution time: %f \n", calculateAverage(pa_tab, attempt));
    printf("Transposition + Parallel execution time: %f \n", calculateAverage(trans_par_tab, attempt));
    saveTabToFile(synchro_tab, attempt, "/home/lshadown/Projects/issfResult/2mmm_synchro.txt");
    saveTabToFile(pa_tab, attempt,  "/home/lshadown/Projects/issfResult/syr2k_pa.txt");
    saveTabToFile(trans_par_tab, attempt, "/home/lshadown/Projects/issfResult/syr2k_trans_par.txt");

}

int main(int argc, char** argv) {
    //gemever(3);
    //symm(3); -- not work
    //syr2k(3);
    _2mm(3);




    return 0;
}


