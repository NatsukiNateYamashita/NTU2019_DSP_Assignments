#include "../inc/hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define STATE_NUM 6 // A-F
#define OBSRV_NUM 6 // A-F
#define SEQ_NUM 50  // 50 observe in a line
#define FNAME_BUF 256  // filename buffer size
#define MAX_TRAIN_DATA_LINE 10000

double a[SEQ_NUM][MAX_STATE]; // forward variable alpha
double b[SEQ_NUM][MAX_STATE]; // backward variable beta
double _gamma[SEQ_NUM][STATE_NUM] = {0}; // cos gamma is used by math.h
double epsilon[SEQ_NUM - 1][STATE_NUM][STATE_NUM] = {0};
double observ_gamma[STATE_NUM][STATE_NUM] = {0}; // for calc
int train_data[MAX_TRAIN_DATA_LINE][SEQ_NUM]; //hmm->observation

void getEpsilon( HMM *hmm, int *O, double epsilon[SEQ_NUM - 1][STATE_NUM][STATE_NUM]){
    double(*A)[MAX_STATE] = hmm->transition;
    double(*B)[MAX_STATE] = hmm->observation;
    int o;
    for ( int t = 0; t < SEQ_NUM - 1; t++){
      double p = 0;
      for ( int i = 0; i < STATE_NUM; i++){
        for ( int j = 0; j < STATE_NUM; j++){
          o = O[t + 1];
          epsilon[t][i][j] = a[t][i] * A[i][j] * B[o][j] * b[t + 1][j];
          p += epsilon[t][i][j];
        }
      }
      for ( int i = 0; i < STATE_NUM; i++){
        for ( int j = 0; j < STATE_NUM; j++){
          epsilon[t][i][j] /= p;
        }
      }
    }
}

void getGamma( int *O, double _gamma[SEQ_NUM][STATE_NUM]){
    for ( int t = 0; t < SEQ_NUM; t++){
      double p = 0;
      for ( int i = 0; i < STATE_NUM; i++){
        _gamma[t][i] = a[t][i] * b[t][i];
        p += _gamma[t][i];
      }
      for ( int i = 0; i < STATE_NUM; i++){
        _gamma[t][i] /= p;
      }
    }
    // calc number of observation O=k in state i
    // for train and update
    for ( int t = 0; t < SEQ_NUM; t++){
      for ( int i = 0; i < STATE_NUM; i++){
        observ_gamma[O[t]][i] += _gamma[t][i];
      }
    }
}

void backward( HMM *hmm, int *O){
    //backward var: b[SEQ_NUM][MAX_STATE]
    double(*A)[MAX_STATE] = hmm->transition;
    double(*B)[MAX_STATE] = hmm->observation;
    for ( int i = 0; i < STATE_NUM; i++){
      b[SEQ_NUM - 1][i] = 1.0;
    }
    for ( int t = SEQ_NUM - 2; t >= 0; t--){
      for ( int i = 0; i < STATE_NUM; i++){
        for ( int j = 0; j < STATE_NUM; j++){
          b[t][i] += A[i][j] * B[O[t + 1]][j] * b[t + 1][j];
        }
      }
    }
}

void forward( HMM *hmm, int *O){
    //forward var: a[SEQ_NUM][MAX_STATE]
    double *pi = hmm->initial;
    double(*A)[MAX_STATE] = hmm->transition;
    double(*B)[MAX_STATE] = hmm->observation;

    for ( int i = 0; i < STATE_NUM; i++){
      a[0][i] = pi[i] * B[O[0]][i];
    }
    for ( int t = 1; t < SEQ_NUM; t++){
      for ( int j = 0; j < STATE_NUM; j++){
        double state_p = 0;
        for ( int i = 0; i < STATE_NUM; i++){
          state_p += a[t - 1][i] * A[i][j];
        }
        a[t][j] = state_p * B[O[t]][j];
      }
    }
}

void train( HMM *hmm, int iterations){
    for ( int iter = 0; iter < iterations; iter++){
      double gamma_sum[SEQ_NUM][STATE_NUM] = {0};
      double epsilon_sum[STATE_NUM][STATE_NUM] = {0};
      memset( observ_gamma, 0, sizeof( observ_gamma));

      // accumulate a iter
      for ( int l = 0; l < MAX_TRAIN_DATA_LINE; l++){
        // reset
        memset( a, 0, sizeof(a));
        memset( b, 0, sizeof(b));
        memset( _gamma, 0, sizeof( _gamma));
        memset( epsilon, 0, sizeof( epsilon));

        forward( hmm, train_data[l]);
        backward( hmm, train_data[l]);
        getGamma( train_data[l], _gamma);
        getEpsilon( hmm, train_data[l], epsilon);

        // accumulate epsilon and gamma
        for ( int t = 0; t < SEQ_NUM; t++){
          for ( int i = 0; i < STATE_NUM; i++){
            gamma_sum[t][i] += _gamma[t][i];
          }
        }
        for ( int t = 0; t < SEQ_NUM - 1; t++){
          for ( int i = 0; i < STATE_NUM; i++){
            for ( int j = 0; j < STATE_NUM; j++){
              epsilon_sum[i][j] += epsilon[t][i][j];
            }
          }
        }
      }
      // update pi
      for ( int i = 0; i < STATE_NUM; i++){
          hmm->initial[i] = gamma_sum[0][i] / MAX_TRAIN_DATA_LINE;
      }
      // gamma(i) is used in update transition and observation
      double gamma_i[STATE_NUM] = {0};
      for ( int t = 0; t < SEQ_NUM - 1; t++){
          for ( int i = 0; i < STATE_NUM; i++){
              gamma_i[i] += gamma_sum[t][i];
          }
      }
      // update transition
      for ( int i = 0; i < STATE_NUM; i++){
        for ( int j = 0; j < STATE_NUM; j++){
          hmm->transition[i][j] = epsilon_sum[i][j] / gamma_i[i];
        }
      }
      // update observation
      for ( int i = 0; i < STATE_NUM; i++){
        gamma_i[i] += gamma_sum[SEQ_NUM-1][i];
      }
      for ( int i = 0; i < STATE_NUM; i++){
        for ( int k = 0; k < OBSRV_NUM; k++){
          hmm->observation[k][i] = observ_gamma[k][i] / gamma_i[i];
        }
      }
    }
}


int main( int argc, char *argv[]){

	int iterations;
	char model_init[FNAME_BUF], train_seq[FNAME_BUF], output_fn[FNAME_BUF];

	if( argc != 5 ){
		printf( "Usage: train_iterations model_init.txt training_sequence_01.txt output_name\n");
		exit(-1);
	}else{
		iterations = atoi( argv[1] );
		strcpy( model_init, argv[2] );
		strcpy( train_seq, argv[3] );
		strcpy( output_fn, argv[4] );
	}

	// printf( "
  //   iterations=%d,
  //   model_init=%s,
  //   train_seq=%s,
  //   output_fn=%s\n",
  //   iterations, model_init, train_seq, output_fn
  // );

  FILE *fp_in, *fp_out;
  char buf[SEQ_NUM + 1];
  fp_in = fopen( train_seq, "r");
  for ( int i = 0; i < MAX_TRAIN_DATA_LINE; i++){
    fscanf( fp_in, "%s", buf);
    for ( int j = 0; j < SEQ_NUM; j++){
      train_data[i][j] = buf[j] - 'A';  //train_data[i][j] has a number representating the char. ex. A->0 B->1
    }
  }
  fclose( fp_in);

  HMM hmm_init;
  loadHMM( &hmm_init, model_init);

  // printf("training iteration : %d\n",iterations);
  train( &hmm_init, iterations);

  fp_out = fopen( output_fn, "w");
  dumpHMM( fp_out, &hmm_init);

  return 0;
}
