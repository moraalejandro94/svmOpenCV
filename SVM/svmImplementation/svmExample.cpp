#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <sys/stat.h>
#include <sys/time.h>
#include <sstream>
#include <string> 
#include <fstream>

using namespace std;
using namespace cv; 
using namespace cv::ml;

#define NUM_CLASSES		4
#define	NUM_BINARY_CLASSIFIERS	((NUM_CLASSES * (NUM_CLASSES-1))/2)

#define BANDS 		128
#define	ROWS		472
#define	COLUMNS		402
String modelName = "Op25C2";                             

#define PIXELS  (ROWS*COLUMNS)

float image_in[PIXELS][BANDS];
float w_vector[BANDS][NUM_BINARY_CLASSIFIERS];
float distances[PIXELS][NUM_BINARY_CLASSIFIERS];
float distancesOpCV[PIXELS][NUM_BINARY_CLASSIFIERS];
float probA[NUM_BINARY_CLASSIFIERS]; 
float probB[NUM_BINARY_CLASSIFIERS];
float rho[NUM_BINARY_CLASSIFIERS];
Mat pixels[PIXELS];
float min_prob = 0.0000001;
float max_prob = 0.9999999;
float sigmoidPredictions [PIXELS][NUM_BINARY_CLASSIFIERS];
float pairwiseProbability[NUM_CLASSES][NUM_CLASSES];
float multiProbabilityQ[NUM_CLASSES][NUM_CLASSES];
float multiProbabilityQP[NUM_CLASSES];
cv::Ptr<cv::ml::SVM> classifiers[NUM_BINARY_CLASSIFIERS];
float prob_estimates[NUM_CLASSES];
float prob_estimates_result[PIXELS][NUM_CLASSES];
float prob_estimates_result_ordered[PIXELS][NUM_CLASSES];
int label[NUM_CLASSES];
int predicted_labels[PIXELS];
int reader;
double read_time = 0;
double processing_time = 0;
double writing_time = 0;
double total_time = 0;
struct timeval tv1, tv2, tv1mult, tv2mult, tv1tot, tv2tot;
Mat pixelMat =  Mat(PIXELS, BANDS, CV_32F);
bool useMats = true;
float pQp;
float diff_pQp;
float max_error, max_error_aux;
float epsilon = 0.005/NUM_CLASSES;

String inputPath = "./inputFiles/" + modelName +"/";
String outputPath = "./outputFiles/" + modelName +"/";
const char *xmlHeader = "<?xml version=\"1.0\"?>\n \
<opencv_storage>\n \
 <opencv_ml_svm>\n \
  <svmType>C_SVC</svmType>\n \
  <kernel>\n \
    <type>LINEAR</type></kernel>\n \
  <C>1.</C>\n \
  <term_criteria><iterations>100</iterations></term_criteria>\n \
  <var_count>128</var_count>\n \
  <class_count>2</class_count>\n \
  <class_labels type_id=\"opencv-matrix\">\n \
    <rows>2</rows>\n \
    <cols>1</cols>\n \
    <dt>i</dt>\n \
    <data>\n \
      -1 1</data></class_labels>\n \
  <sv_total>1</sv_total>\n \
  <support_vectors>\n \
    <_>\n";

const char *xmlPart2 = "</_></support_vectors> \n \
  <uncompressed_sv_total>0</uncompressed_sv_total> \n \
  <decision_functions> \n \
    <_> \n \
      <sv_count>1</sv_count> \n \
      <rho> ";

const char *xmlEnd = "</rho>  \n \
      <alpha>  \n \
        1.</alpha>  \n \
      <index>  \n \
        0</index></_></decision_functions></opencv_ml_svm>  \n \
</opencv_storage>  \n";
 

cv::Ptr<cv::ml::SVM> initDefaultSVM(){
    cv::Ptr<cv::ml::SVM> svm;
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);    
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    return svm;
}

float vectorProduct(int size, int classIndex, int pixelIndex){
    float product;
    for(int i=0; i<size; i++){
        product += image_in[pixelIndex][i] * w_vector[i][classIndex];
    }
    return product;
}

void setDistanceToVector(int pixelIndex, int svmIndex){      
    if(useMats){
        float distance = classifiers[svmIndex]->predict(pixels[pixelIndex], cv::noArray(),cv::ml::StatModel:: RAW_OUTPUT);
        distances[pixelIndex][svmIndex] = distance;   
        distancesOpCV [pixelIndex][svmIndex] = distance;   
    }  
    else {
        float distance = 0;
        for(int i=0; i<BANDS; i++){
            distance += image_in[pixelIndex][i] * w_vector[i][svmIndex];
        }        
        distance -= rho[svmIndex];
        distances[pixelIndex][svmIndex] = distance;
        int count = 0;       
    }
    
}

void getSigmoidPrediction(int pixelIndex, int svmIndex){
    float sigmoidPrediction;    
    float sigmoid_prediction_fApB = distances[pixelIndex][svmIndex]*probA[svmIndex] + probB[svmIndex];
    if (sigmoid_prediction_fApB >= 0.0){
        sigmoidPrediction = exp(-sigmoid_prediction_fApB)/(1.0+exp(-sigmoid_prediction_fApB));
    }
    else{
        sigmoidPrediction = 1.0/(1.0+exp(sigmoid_prediction_fApB));
    }
    if(sigmoidPrediction < min_prob){
        sigmoidPrediction = min_prob; 
    }		
    if(sigmoidPrediction > max_prob){
        sigmoidPrediction = max_prob;
    }    
    sigmoidPredictions[pixelIndex][svmIndex] = sigmoidPrediction;
}

void getClassProbability(int pixelIndex){ 
    ///////////////////////////////////////Probabilidades Binarias para cada clase //////////////////////////////////////////////////////    
    int svmIndex = 0;             
    for (int p = 0; p < NUM_CLASSES - 1; p++){  
        for (int q = p+1; q < NUM_CLASSES ; q++){                                                 
            pairwiseProbability[p][q] = sigmoidPredictions[pixelIndex][svmIndex];            
            pairwiseProbability[q][p] = 1-pairwiseProbability[p][q];                	                 
            svmIndex ++;
        }              
    }

    ///////////////////////////////////////Probabilidades Multplies para cada clase //////////////////////////////////////////////////////
    for(int b=0; b<NUM_CLASSES; b++){             
			prob_estimates[b] = 1.0/NUM_CLASSES;
			multiProbabilityQ[b][b] = 0.0;
			for(int j=0; j<b; j++){				
                multiProbabilityQ[b][b] += pairwiseProbability[j][b] * pairwiseProbability[j][b];
				multiProbabilityQ [b][j] = multiProbabilityQ[j][b];                
			}
			for(int j=b+1; j<NUM_CLASSES; j++){
				multiProbabilityQ[b][b] += pairwiseProbability[j][b] * pairwiseProbability[j][b];
				multiProbabilityQ[b][j] = -pairwiseProbability[j][b] * pairwiseProbability[b][j];                
			}
	}

    int iters = 0;
    bool stop = false;

    while (!stop || iters == 100){ 
        pQp = 0.0;
        for(int b=0; b<NUM_CLASSES; b++){
            multiProbabilityQP[b] = 0.0;
            for(int j=0; j<NUM_CLASSES; j++){
                multiProbabilityQP[b] += multiProbabilityQ[b][j] * prob_estimates[j];
            }
            pQp += prob_estimates[b] * multiProbabilityQP[b];

        }
        max_error = 0.0;
        for(int b=0; b<NUM_CLASSES; b++){
            max_error_aux = multiProbabilityQP[b] - pQp; // Same as ^2 then sqrt(?)
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
        if(!stop){
            for(int b=0; b<NUM_CLASSES; b++){
                diff_pQp = (-multiProbabilityQP[b]+pQp)/(multiProbabilityQ[b][b]);
                prob_estimates[b] = prob_estimates[b] + diff_pQp;
                pQp = ((pQp + diff_pQp * (diff_pQp*multiProbabilityQ[b][b]+2*multiProbabilityQP[b]))/(1+diff_pQp))/(1+diff_pQp);
                for(int j=0; j<NUM_CLASSES; j++){
                    multiProbabilityQP[j] = (multiProbabilityQP[j] + diff_pQp*multiProbabilityQ[b][j])/(1+diff_pQp);
                    prob_estimates[j] = prob_estimates[j]/(1+diff_pQp);
                }
            }
        }
        iters++;        
    }
    for(int b=0; b<NUM_CLASSES; b++){
        prob_estimates_result[pixelIndex][b] = prob_estimates[b];
    }
}


double timeval_diff(struct timeval *tfin, struct timeval *tini){
	return (double)(tfin->tv_sec + (double)tfin->tv_usec/1000000) -(double)(tini->tv_sec + (double)tini->tv_usec/1000000);
}


void decision(int pixelIndex){
    int decision = 0;
		for(int b=1; b<NUM_CLASSES; b++){
			if(prob_estimates[b] > prob_estimates[decision]){
				decision = b;
			}            
		}
		
		predicted_labels[pixelIndex] = label[decision];
        
    for(int j=0; j<NUM_CLASSES; j++){
        int position = label[j]-1;
        prob_estimates_result_ordered[pixelIndex][position] = prob_estimates_result[pixelIndex][j];
    }
        
}

void svmMain(){
    gettimeofday(&tv1,NULL);
    for(int i = 0; i < PIXELS; i++){
        for(int j = 0; j < NUM_BINARY_CLASSIFIERS ; j++){
            setDistanceToVector(i, j);
            getSigmoidPrediction(i,j);
        }        
    } 
    for(int i = 0; i < PIXELS; i++){
        getClassProbability(i);
        decision(i);
    }
    gettimeofday(&tv2,NULL);
	processing_time = timeval_diff(&tv2, &tv1);
}

void readFiles(){    
    FILE *fp;

    gettimeofday(&tv1,NULL);
    String imgPathStr = inputPath  + "image.bin";
    const char *imgPath = imgPathStr.c_str();
	fp = fopen(imgPath,"rb");
    if (fp == NULL)
    {
        printf("FAIL!! \n");
        exit(EXIT_FAILURE);
    }
    float currBand; 
	for(int j=0; j < PIXELS; j++){
        Mat currMat = Mat(1, BANDS, CV_32F);        
        for(int k=0; k<BANDS; k++){            
            // Create Pixel Mat     	            
            reader = fread(&currBand, sizeof(float), 1, fp);		
            //reader = fread(&currMat.at<float>(Point(0,k)), sizeof(float), 1, fp);                        
            currMat.at<float>(0, k) = currBand;
		}
        pixels[j] = currMat;
	}

	fclose(fp);
    String probAPathStr = inputPath  + "ProbA.bin";
    const char *probAPath = probAPathStr.c_str();
	fp = fopen(probAPath,"rb");
	for(int j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&probA[j], sizeof(float), 1, fp);
	}
	fclose(fp);
    String probBPathStr = inputPath  + "ProbB.bin";
    const char *probBPath = probBPathStr.c_str();
	fp = fopen(probBPath,"rb");
	for(int j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&probB[j], sizeof(float), 1, fp);
	}
    fclose(fp);
    String labelPathStr = inputPath  + "label.bin";
    const char *labelPath = labelPathStr.c_str();
    fp = fopen(labelPath,"rb");
	for(int j=0; j < NUM_CLASSES; j++){
		reader = fread(&label[j], sizeof(int), 1, fp);
	}	
	fclose(fp);
    

	gettimeofday(&tv2,NULL);
	read_time = timeval_diff(&tv2, &tv1);
}

void writeFiles(){
    FILE *fp;
    gettimeofday(&tv1,NULL);

    String prob_estimatesPathStr = outputPath  + "implementationProb.txt";
    const char *prob_estimatesPath = prob_estimatesPathStr.c_str();
	fp = fopen(prob_estimatesPath,"w"); //Fichero de salida
	if(fp==NULL){
		printf("open output matrix_mult failed\n");
	}else{
		for(int i = 0; i < PIXELS; i++){
			for (int j = 0; j < NUM_CLASSES; j++){
				fprintf(fp,"%.9f\n",prob_estimates_result_ordered[i][j]);
			}
		}
	}
	fclose(fp);

    String labels_obtainedPathStr = outputPath  + "implementationLabels.txt";
    const char *labels_obtainedPath = labels_obtainedPathStr.c_str();
	fp = fopen(labels_obtainedPath,"w"); //Fichero de salida
	if(fp==NULL){
		printf("open output matrix_mult failed\n");
	}else{
		for(int i = 0; i < PIXELS; i++){
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

void writeTimes(){
    FILE *fp;
    String timesPathStr = outputPath  + "times.txt";
    const char *timesPath = timesPathStr.c_str();
	fp = fopen(timesPath,"w"); //Fichero de salida
	if(fp==NULL){
		printf("open output matrix_mult failed\n");
	}else{		
        fprintf(fp,"Reading: %.4g ms.... Processing: %.4g ms.... Writing: %.4g ms\n", read_time*1000.0, processing_time*1000.0,writing_time*1000.0);		
        fprintf(fp,"TOTAL TIME: %.4g ms\n", total_time*1000.0);
	}
    fclose(fp);

}

void writeInputFiles(){
    FILE *fp;    
    String rhoPathStr = inputPath  + "rho.bin";
    const char *rhoPath = rhoPathStr.c_str();
    fp = fopen(rhoPath,"rb");
	for(int j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&rho[j], sizeof(float), 1, fp);
	}

    String w_vectorPathStr = inputPath  + "w_vector.bin";
    const char *w_vectorPath = w_vectorPathStr.c_str();
    fp = fopen(w_vectorPath,"rb");
	for(int j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		for(int k=0; k < BANDS; k++){
			reader = fread(&w_vector[k][j], sizeof(float), 1, fp);                    
		}
	}
	fclose(fp);


    for(int i = 0; i <NUM_BINARY_CLASSIFIERS; i++){        
        std::stringstream ss;
        ss << "./inputFiles/"<<modelName<<"/svmBands" << i << ".xml";        
        std::string fileName = ss.str(); 
        fp = fopen(fileName.c_str(),"w"); //Fichero de salida
        if(fp==NULL){
            printf("open output matrix_mult failed\n");
        }else{            
            fprintf(fp,"%s",xmlHeader);
            for (int j = 0; j < BANDS; j++){
                fprintf(fp,"%.9f\n",w_vector[j][i]);
            }
            fprintf(fp,"%s",xmlPart2);
            fprintf(fp,"%.9f\n",rho[i]);
            fprintf(fp,"%s",xmlEnd);                            
        }
        fclose(fp);    
    }
}

void buildSVM(){    
    for (int i = 0; i < NUM_BINARY_CLASSIFIERS; i++){
        std::stringstream ss;
        ss << "./inputFiles/"<< modelName <<"/svmBands" << i << ".xml";        
        std::string fileName = ss.str(); 
        cv::Ptr<cv::ml::SVM> svm;
        svm = initDefaultSVM();
        svm = Algorithm::load<SVM>(fileName);    //Load SVM from file
        classifiers[i] = svm;
    }
}

int main(int, char**)
{    
    bool useAlgorithm = true;
    if (useAlgorithm){
        gettimeofday(&tv1tot,NULL);
        readFiles();
        writeInputFiles();
        buildSVM();
        svmMain();        
        writeFiles();   
        writeTimes();     
    }
    else{
        //svmExample();        
    }      
}