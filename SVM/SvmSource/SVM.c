#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include <float.h>
#include <sys/stat.h>
#include <sys/time.h>

//DEFINICIÓN DE LOS PARÁMETROS DE ENTRADA DE LA IMAGEN

#define NUM_CLASSES		4
#define	NUM_BINARY_CLASSIFIERS	((NUM_CLASSES * (NUM_CLASSES-1))/2)

#define BANDS 		128
#define	ROWS		459
#define	COLUMNS		548

#define PIXELS  (ROWS*COLUMNS)
int i, j, k, q, b, p, clasificador;

float image_in[PIXELS][BANDS];
int label[NUM_CLASSES];
float probA[NUM_BINARY_CLASSIFIERS]; 
float probB[NUM_BINARY_CLASSIFIERS];
float rho[NUM_BINARY_CLASSIFIERS];
float w_vector[BANDS][NUM_BINARY_CLASSIFIERS];

float sum1;
float dec_values[NUM_BINARY_CLASSIFIERS];
float sigmoid_prediction_fApB;
float sigmoid_prediction;
float min_prob = 0.0000001;
float max_prob = 0.9999999;
float pairwise_prob[NUM_CLASSES][NUM_CLASSES];
float multi_prob_Q[NUM_CLASSES][NUM_CLASSES];
int iters, stop;
float pQp;
float multi_prob_Qp[NUM_CLASSES];
float max_error, max_error_aux;
float epsilon = 0.005/NUM_CLASSES;
float diff_pQp;
int decision;
float prob_estimates[NUM_CLASSES];
float prob_estimates_result[PIXELS][NUM_CLASSES];
float prob_estimates_result_ordered[PIXELS][NUM_CLASSES];
int position;
int predicted_labels[PIXELS];

double read_time = 0;
double processing_time = 0;
double writing_time = 0;
double total_time = 0;

int reader;

double timeval_diff(struct timeval *tfin, struct timeval *tini){
	return (double)(tfin->tv_sec + (double)tfin->tv_usec/1000000) -(double)(tini->tv_sec + (double)tini->tv_usec/1000000);
}

int main()
{	
	struct timeval tv1, tv2, tv1mult, tv2mult, tv1tot, tv2tot;
	FILE *fp;
	
	gettimeofday(&tv1tot,NULL);
	
	//LECTURA DE FICHEROS
	gettimeofday(&tv1,NULL);
	fp = fopen("image.bin","rb");
	for(j=0; j < PIXELS; j++){
		for(k=0; k<BANDS; k++){
			reader = fread(&image_in[j][k], sizeof(float), 1, fp);
		}
	}
	fclose(fp);
	fp = fopen("label.bin","rb");
	for(j=0; j < NUM_CLASSES; j++){
		reader = fread(&label[j], sizeof(int), 1, fp);
	}
	fclose(fp);
	fp = fopen("ProbA.bin","rb");
	for(j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&probA[j], sizeof(float), 1, fp);
	}
	fclose(fp);
	fp = fopen("ProbB.bin","rb");
	for(j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&probB[j], sizeof(float), 1, fp);
	}
	fclose(fp);
	fp = fopen("rho.bin","rb");
	for(j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&rho[j], sizeof(float), 1, fp);
	}
	fclose(fp);
	fp = fopen("w_vector.bin","rb");
	for(j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		for(k=0; k < BANDS; k++){
			reader = fread(&w_vector[k][j], sizeof(float), 1, fp);
		}
	}
	fclose(fp);

	gettimeofday(&tv2,NULL);

	read_time = timeval_diff(&tv2, &tv1);

	//CLASIFICACIÓN
	gettimeofday(&tv1,NULL);

	for(q=0; q<PIXELS; q++){		//para cada pixel
		p = 0;
		clasificador = 0;
		for(b=0; b<NUM_CLASSES; b++){		//para todas las combinaciones de clases (clasificadores binarios) calculamos las probabilidades binarias
			for(k=b+1; k<NUM_CLASSES; k++){
				sum1 = 0.0;
				for(j=0; j<BANDS; j++){
					sum1 += image_in[q][j] * w_vector[j][clasificador];
				}
				dec_values[clasificador] = sum1 - rho[p];				
				sigmoid_prediction_fApB = dec_values[clasificador]*probA[p] + probB[p];

				if (sigmoid_prediction_fApB >= 0.0){
					sigmoid_prediction = exp(-sigmoid_prediction_fApB)/(1.0+exp(-sigmoid_prediction_fApB));
				}
				else{
					sigmoid_prediction = 1.0/(1.0+exp(sigmoid_prediction_fApB));
				}
				if(sigmoid_prediction < min_prob){
					sigmoid_prediction = min_prob;
				}		
				if(sigmoid_prediction > max_prob){
					sigmoid_prediction = max_prob;
				}							
				pairwise_prob[b][k] = sigmoid_prediction;
				pairwise_prob[k][b] = 1-sigmoid_prediction;					
				p++;
				clasificador++;
			}
		}
		///////////////////////// Multi Prob //////////////////////////////////////
		p = 0;
		for(b=0; b<NUM_CLASSES; b++){ //para todos los clasificadores binarios
			prob_estimates[b] = 1.0/NUM_CLASSES;
			multi_prob_Q[b][b] = 0.0;
			for(j=0; j<b; j++){
				multi_prob_Q[b][b] += pairwise_prob[j][b] * pairwise_prob[j][b];
				multi_prob_Q[b][j] = multi_prob_Q[j][b];					
			}
			for(j=b+1; j<NUM_CLASSES; j++){
				multi_prob_Q[b][b] += pairwise_prob[j][b] * pairwise_prob[j][b];
				multi_prob_Q[b][j] = -pairwise_prob[j][b] * pairwise_prob[b][j];
				if (q < 10){
					printf("Pixel:  %d", q );
					printf("  b:  %d", b );
					printf("  j:  %d", j );
					printf("  MultiProb BB %f  ", multi_prob_Q[b][b]);	
					printf("  MultiProb BJ %f \n", multi_prob_Q[b][q]);	
				}	
			}								
		}

		iters = 0;
		stop = 0;

		while (stop == 0){ // OR iters == 100) and remove lines 197-199
				
			pQp = 0.0;
			for(b=0; b<NUM_CLASSES; b++){
				multi_prob_Qp[b] = 0.0;
				for(j=0; j<NUM_CLASSES; j++){
					multi_prob_Qp[b] += multi_prob_Q[b][j] * prob_estimates[j];
				}
				pQp += prob_estimates[b] * multi_prob_Qp[b];
			}
			max_error = 0.0;
			for(b=0; b<NUM_CLASSES; b++){
				max_error_aux = multi_prob_Qp[b] - pQp; // Same as ^2 then sqrt(?)
				if (max_error_aux < 0.0){
					max_error_aux = -max_error_aux;
				}
				if (max_error_aux > max_error){
					max_error = max_error_aux;
				}
			}
			if(max_error < epsilon){
				stop = 1;
			}
			if(stop == 0){
				for(b=0; b<NUM_CLASSES; b++){
					diff_pQp = (-multi_prob_Qp[b]+pQp)/(multi_prob_Q[b][b]);
					prob_estimates[b] = prob_estimates[b] + diff_pQp;
					pQp = ((pQp + diff_pQp * (diff_pQp*multi_prob_Q[b][b]+2*multi_prob_Qp[b]))/(1+diff_pQp))/(1+diff_pQp);
					for(j=0; j<NUM_CLASSES; j++){
						multi_prob_Qp[j] = (multi_prob_Qp[j] + diff_pQp*multi_prob_Q[b][j])/(1+diff_pQp);
						prob_estimates[j] = prob_estimates[j]/(1+diff_pQp);
					}
				}
			}
			iters++;
			
			if(iters == 100){
				stop = 1;
			}
		}

		for(b=0; b<NUM_CLASSES; b++){
			prob_estimates_result[q][b] = prob_estimates[b];
		}
		
		//elección
		decision = 0;
		for(b=1; b<NUM_CLASSES; b++){
			if(prob_estimates[b] > prob_estimates[decision]){
				decision = b;
			}
			if (q % 10000 == 0){                
				
            }
		}
		
		predicted_labels[q] = label[decision];
	}

	for(i=0; i<PIXELS; i++){
		for(j=0; j<NUM_CLASSES; j++){
			position = label[j]-1;
			prob_estimates_result_ordered[i][position] = prob_estimates_result[i][j];
		}
	}

	gettimeofday(&tv2,NULL);

	processing_time = timeval_diff(&tv2, &tv1);

	//ESCRITURA DEL RESULTADO
	gettimeofday(&tv1,NULL);

	fp = fopen("prob_estimates.txt","w"); //Fichero de salida
	if(fp==NULL){
		printf("open output matrix_mult failed\n");
	}else{
		for(i = 0; i < PIXELS; i++){
			for (j = 0; j < NUM_CLASSES; j++){
				fprintf(fp,"%.9f\n",prob_estimates_result_ordered[i][j]);
			}
		}
	}
	fclose(fp);

	fp = fopen("labels_obtained.txt","w"); //Fichero de salida
	if(fp==NULL){
		printf("open output matrix_mult failed\n");
	}else{
		for(i = 0; i < PIXELS; i++){
			fprintf(fp,"%d\n",predicted_labels[i]);
		}
	}
	fclose(fp);

	gettimeofday(&tv2,NULL);

	writing_time = timeval_diff(&tv2, &tv1);

	gettimeofday(&tv2tot,NULL);
	total_time = timeval_diff(&tv2tot, &tv1tot);

	printf("Reading: %.4g ms.... Processing: %.4g ms.... Writing: %.4g ms\n", read_time*1000.0, processing_time*1000.0,writing_time*1000.0);
	printf("TOTAL TIME: %.4g ms\n", total_time*1000.0);

}
