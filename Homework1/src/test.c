#include "../inc/hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define MAX_MODEL_NUM 5
#define SEQ_NUM 50
#define STATE_NUM 6
#define FNAME_BUF 256 // filename buffer size
#define ANSWER_MODEL_NAME_BUF 13
#define MAX_TEST_DATA_LINE 2500

char answer_model_name[MAX_TEST_DATA_LINE][FNAME_BUF];
double answer_probability[MAX_TEST_DATA_LINE];

void test( HMM *hmm, int test_data[MAX_TEST_DATA_LINE][SEQ_NUM]){
	for ( int d = 0; d < MAX_TEST_DATA_LINE; d++){
		int *O = test_data[d];
		double model_max = DBL_MIN;
		char model_name[FNAME_BUF] = {0};

		for ( int m = 0; m < MAX_MODEL_NUM; m++){
			double delta[SEQ_NUM][STATE_NUM] = {0};
			double(*A)[MAX_STATE] = hmm[m].transition;
			double(*B)[MAX_STATE] = hmm[m].observation;

			// initialization
			for ( int i = 0; i < STATE_NUM; i++){
				delta[0][i] = hmm[m].initial[i] * B[O[0]][i];
			}

			// recursion
			for ( int t = 1; t < SEQ_NUM; t++){
				for ( int j = 0; j < STATE_NUM; j++){
					double max = DBL_MIN;
					for ( int i = 0; i < STATE_NUM; i++){
						double temp_max = delta[t - 1][i] * A[i][j];
						if ( temp_max > max)
							max = temp_max;
					}
					delta[t][j] = max * B[O[t]][j];
				}
			}

			// termination
			double max = DBL_MIN;
			for ( int i = 0; i < STATE_NUM; i++){
				if ( delta[SEQ_NUM - 1][i] > max)
					max = delta[SEQ_NUM - 1][i];
			}

      // update max probability
			if ( max > model_max){
				model_max = max;
				snprintf( model_name, ANSWER_MODEL_NAME_BUF, "model_0%d.txt", m + 1);
			}

		}

		strncpy( answer_model_name[d], model_name, ANSWER_MODEL_NAME_BUF);
		answer_probability[d] = model_max;
	}
}

int main( int argc, char *argv[]){

  char modellist[FNAME_BUF], test_seq[FNAME_BUF], output_fn[FNAME_BUF];

  if( argc != 4 ){
    printf( "Usage: modellist.txt test_seq.txt output_name\n");
    exit(-1);
  }else{
    strcpy( modellist, argv[1] );
    strcpy( test_seq, argv[2] );
    strcpy( output_fn, argv[3] );
  }

  // printf("modellist=%s,test_seq=%s,output_fn=%s\n",
  //   modellist, test_seq, output_fn);

	int test_data[MAX_TEST_DATA_LINE][SEQ_NUM];
	FILE *fp_in, *fp_out;
  char buf[SEQ_NUM + 1];
	fp_in = fopen( test_seq, "r");
	for ( int i = 0; i < MAX_TEST_DATA_LINE; i++){
		fscanf( fp_in, "%s", buf);
		for ( int j = 0; j < SEQ_NUM; j++){
			test_data[i][j] = buf[j] - 'A';
		}
	}
	fclose(fp_in);

  // printf("test_data");

	HMM hmms[MAX_MODEL_NUM];
	load_models( modellist, hmms, MAX_MODEL_NUM);
	test( hmms, test_data);

	fp_out = fopen(output_fn, "w");
	for ( int i = 0; i < MAX_TEST_DATA_LINE; i++){
		fprintf( fp_out, "%s %.10e\n", answer_model_name[i], answer_probability[i]);
	}

	return 0;
}
