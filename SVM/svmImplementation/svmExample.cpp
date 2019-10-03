#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <sys/stat.h>
#include <sys/time.h>

using namespace cv;
using namespace cv::ml;

#define NUM_CLASSES		4
#define	NUM_BINARY_CLASSIFIERS	((NUM_CLASSES * (NUM_CLASSES-1))/2)

#define BANDS 		128
#define	ROWS		459
#define	COLUMNS		548

#define PIXELS  (ROWS*COLUMNS)

float image_in[PIXELS][BANDS];
float w_vector[BANDS][NUM_BINARY_CLASSIFIERS];
float distances[PIXELS][NUM_BINARY_CLASSIFIERS];
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
bool useMats = false;
float pQp;
float diff_pQp;
float max_error, max_error_aux;
float epsilon = 0.005/NUM_CLASSES;
String modelName = "Op81C";                             //Nombre del modelo 
String inputPath = "./inputFiles/" + modelName +"/";
String outputPath = "./outputFiles/" + modelName +"/";
 

cv::Ptr<cv::ml::SVM> initDefaultSVM(int sampleCount,int sampleSize){
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
                if (pixelIndex < 10){
					printf("Pixel:  %d", pixelIndex );
					printf("  b:  %d", b );
					printf("  j:  %d", j );
					printf("  MultiProb BB %f  ", multiProbabilityQ[b][b]);	
					printf("  MultiProb BJ %f \n", multiProbabilityQ[b][pixelIndex]);	
				}	
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
	for(int j=0; j < PIXELS; j++){
        for(int k=0; k<BANDS; k++){            
            // Create Pixel Mat     	
            reader = fread(&image_in[j][k], sizeof(float), 1, fp);		
            //reader = fread(&pixelMat.at<float>(Point(j,k)), sizeof(float), 1, fp);            
            if (j==1){
                //std::cout << "MAT: " << pixelMat.at<float>(Point(j,k)) << "\n";                
            } 
		}
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
    String rhoPathStr = inputPath  + "rho.bin";
    const char *rhoPath = rhoPathStr.c_str();
    fp = fopen(rhoPath,"rb");
	for(int j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		reader = fread(&rho[j], sizeof(float), 1, fp);
	}
    fclose(fp);
    String labelPathStr = inputPath  + "label.bin";
    const char *labelPath = labelPathStr.c_str();
    fp = fopen(labelPath,"rb");
	for(int j=0; j < NUM_CLASSES; j++){
		reader = fread(&label[j], sizeof(int), 1, fp);
	}	
	fclose(fp);
    String w_vectorPathStr = inputPath  + "w_vector.bin";
    const char *w_vectorPath = w_vectorPathStr.c_str();
    fp = fopen(w_vectorPath,"rb");
	for(int j=0; j < NUM_BINARY_CLASSIFIERS; j++){
		for(int k=0; k < BANDS; k++){
			reader = fread(&w_vector[k][j], sizeof(float), 1, fp);                    
		}
	}
	fclose(fp);

    // Load OpenCV SVM files into SVM array 

	gettimeofday(&tv2,NULL);
	read_time = timeval_diff(&tv2, &tv1);
    
}

void writeFiles(){
    FILE *fp;
    gettimeofday(&tv1,NULL);

    String prob_estimatesPathStr = outputPath  + "prob_estimates.txt";
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

    String labels_obtainedPathStr = outputPath  + "labels_obtained.txt";
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


void svmExample(){
    // Set up training data    
    int sampleCount = 5;
    int sampleSize = 2;
    int labels[sampleCount] = {1, 1,-1, -1, -1};
    float trainingData[sampleCount][sampleSize] = { {350, 200},{501, 10}, {255, 300}, {255, 255}, {300, 420} };
    Mat trainingDataMat(4, 2, CV_32F, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);
    String fileName = "./inputFiles/OpenCV/svmTest.xml";
    
    //Config Values
    bool trainSVM = true;
    bool saveSVM = true;

    // Init SVM
    cv::Ptr<cv::ml::SVM> svm;
    if (trainSVM){
    svm = initDefaultSVM(sampleCount, sampleSize);
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
        if (saveSVM){            
            svm->save(fileName);        
        }    
    }
    else{        
        svm = Algorithm::load<SVM>(fileName);    //Load SVM from file
    } 

    
    int width = 512, height = 512;
    bool returnDFVal = true;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    Vec3b green(0,255,0), blue(255,0,0);
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){                        
            Mat sampleMat = (Mat_<float>(1,2) << j,i);            
            if(int v2 = rand() % 1000 + 1 > 992){
                float response = svm->predict(sampleMat, cv::noArray(),cv::ml::StatModel:: RAW_OUTPUT);                
                if (response > 0){
                    image.at<Vec3b>(i,j)  = Vec3b(0,255-(response*14),-1);
                }
                else{
                    image.at<Vec3b>(i,j)  = Vec3b(255+(response*14),1,0);
                }                                                                
            }        
            else{
                image.at<Vec3b>(i,j)  = Vec3b(0,0,0);
            }    
        }
    }
    int thickness = -1;
    for (int i = 0 ; i < sampleCount ; i++){        
        if (labels[i] == 1){
            circle( image, Point(trainingData[i][0],  trainingData[i][1]), 5, Scalar(  255,   180,   0), thickness);    
        }
        else{
            circle( image, Point(trainingData[i][0],  trainingData[i][1]), 5, Scalar(  0,   180,   255), thickness );    
        }                
    }

    thickness = 2;
    Mat sv = svm->getUncompressedSupportVectors();

    for (int i = 0; i < sv.rows; i++){
        const float* v = sv.ptr<float>(i);
        circle(image,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness);
    }    

    imwrite("./outputFiles/OpenCV/result.png", image);        //Save the image
    imshow("SVM", image);               //Display the image 
    waitKey();
}

int main(int, char**)
{
    bool useAlgorithm = true;
    if (useAlgorithm){
        readFiles();
        svmMain();
        writeFiles();
    }
    else{
        svmExample();
    }
    return 0;
}