#include <iostream>
#include <sys/stat.h>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <fstream>

using namespace std;

#define BANDS 		128
#define	ROWS		459
#define	COLUMNS		548
#define PIXELS  (ROWS*COLUMNS)
#define NUM_CLASSES		4

double sourceProbabilities[PIXELS*NUM_CLASSES];
double implementationProbabilities[PIXELS*NUM_CLASSES];
float impProb[PIXELS*NUM_CLASSES];
int labels[PIXELS];
int labelsObtained[PIXELS];

int sourceLabels[PIXELS];
int implementationLabels[PIXELS];

int reader;

using namespace std;

string modelName = "Op81C";                             //Nombre del modelo 
string openCVPath = "./inputFiles/" + modelName +"/";
string sourcePath = "./outputFiles/" + modelName +"/";


void readSourceLabels(){
    ifstream in("labels_obtained.bin", ios::in | ios::binary);
    in.read((char *) &sourceLabels, sizeof sourceLabels);
    // see how many bytes have been read
    cout << in.gcount() << " bytes read\n";
    in.close();
}

void readFiles(){
    FILE *fp;

    ifstream in("labels", ios::in | ios::binary);
    in.read((char *) &implementationLabels, sizeof implementationLabels);
    // see how many bytes have been read
    cout << in.gcount() << " bytes read\n";
    in.close();

    fp = fopen("sourceProb.txt","r");
    for(int j=0; j < PIXELS*NUM_CLASSES; j++){
		reader = fread(&sourceProbabilities[j], sizeof(double), 1, fp);
	}
	fclose(fp);
    fp = fopen("implementationProb","rb");
    for(int j=0; j < PIXELS*NUM_CLASSES; j++){
		reader = fread(&implementationProbabilities[j], sizeof(double), 1, fp);
	}
	fclose(fp);

    readSourceLabels();
}



int main()
{
    printf("-------------------Probabilities-------------------\n");
    readFiles();
    int probCount = 0;
    double probLargestDiff = 0;
    for (int i = 0 ; i < PIXELS*NUM_CLASSES ; i++){
        double source = sourceProbabilities[i];
        double imp = implementationProbabilities[i];
        if (source != imp){            
            
            if (source - imp > 0 && source - imp > probLargestDiff){
                probLargestDiff = source - imp;
                printf("INDEX %d", i);
                printf("  SOURCE  %f", source);
                printf("  IMPLEMENTATION  %f \n", imp);
                probCount++;
            }
            if (imp - source > 0 && imp - source > probLargestDiff){
                probLargestDiff = imp - source;
                printf("INDEX %d", i);
                printf("  SOURCE  %f", source);
                printf("  IMPLEMENTATION  %f \n", imp);
                probCount++;
            }            
            
        }
    }

    double percentage = probCount*100/PIXELS*NUM_CLASSES;
    printf("TOTAL  %d \n", probCount);
    printf("PERCENTAGE  %f \n", percentage);
    printf("LARGEST DIFF %f \n", probLargestDiff);
    printf("-------------------Labels-------------------\n");
    int labelCount = 0;
    double labelLargestDiff = 0;
    for (int i = 0 ; i < PIXELS ; i++){
        int source = sourceLabels[i];
        int imp = implementationLabels[i];
        if (source != imp){            
            
            if (source - imp > 0){
                labelLargestDiff = source - imp;
                printf("INDEX %d", i);
                printf("  SOURCE  %d", source);
                printf("  IMPLEMENTATION  %d \n", imp);
                labelCount++;
            }
            if (imp - source > 0){
                labelLargestDiff = imp - source;
                printf("INDEX %d", i);
                printf("  SOURCE  %d", source);
                printf("  IMPLEMENTATION  %d \n", imp);
                labelCount++;
            }            
            
        }
    }
    percentage = labelCount*100/PIXELS*NUM_CLASSES;
    printf("TOTAL  %d \n", labelCount);
    printf("PERCENTAGE  %f \n", percentage);
    printf("LARGEST DIFF %f \n", labelLargestDiff);
    
    for(int i = 0; i < 10; i++){
        printf("INDEX %d", i);
        printf(" %d \n", sourceLabels[i]);
    }
    

    
    return 0;
}