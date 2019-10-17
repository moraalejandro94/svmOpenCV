#include<stdio.h> 
#include<string.h> 
#include<stdlib.h> 
#include <string> 
#include <iostream>
float errorThreshold = 0.00109; 
float largestDif = 0;
float largestDifImp = 0;
float largestDifSrc = 0;
int largestDifLine = 0;
int globalErrors = 0;
  
void compareFiles(FILE *fp1, FILE *fp2, bool printDif) 
{ 

    char ch1 = getc(fp1); 
    char ch2 = getc(fp2); 
    bool error = false;
    std::string str1;
    std::string str2;
    largestDif = 0;
    largestDifImp = 0;
    largestDifSrc = 0;
    largestDifLine = 0;

    int errorCount = 0, pos = 0, line = 1; 

    while (ch1 != EOF && ch2 != EOF) 
    {         
        pos++;   
        if (ch1 == '\n' && ch2 == '\n') 
        { 
            if (error){                
                float imp =strtof((str1).c_str(),0);
                float src =strtof((str2).c_str(),0);
                float dif = imp - src;
                if (dif < 0){
                    dif *= -1;
                }
                if (dif > largestDif){
                    largestDif = dif;
                    largestDifImp = imp;
                    largestDifSrc = src;
                    largestDifLine = line;
                }            
                if (dif > errorThreshold){
                    errorCount++;
                    globalErrors++;
                    std::cout<<"Line:  " << line << "  Imp:  " << str1 << "  Src:  " << str2 << "  Dif:  " << dif<<"\n"; 
                }                   
            }
            line++; 
            pos = 0; 
            str1 = "";
            str2 = "";            
            error = false;
        } 
  
        if (ch1 != ch2) 
        {  
            error = true;
            
        } 
        if(ch1 != '\n'){
            str1 += std::string(1, ch1);
        }
        if(ch2 != '\n'){
            str2 += std::string(1, ch2);
        }        
        ch1 = getc(fp1); 
        ch2 = getc(fp2); 
    } 
    printf("Total Errors : %d\n", errorCount); 
    if(printDif){
        std::cout<<"Largest Difference:      Line:  " << largestDifLine << "  Imp:  " << largestDifImp << "  Src:  " << largestDifSrc << "   Dif:  " << largestDif<<"\n";
    }    
} 

void globalTest(){
    int sampleCount = 6;
    std::string modelName = "Op12C2";
    std::string models[sampleCount] = {"Op8C1","Op8C2","Op12C1","Op12C2","Op20C1","Op25C2"};
    for (int i = 0; i < sampleCount; i++){        
        std::string fp1PathStr = "./TestFiles/" + models[i] + "/implementationProb.txt";
        const char *fp1Path = fp1PathStr.c_str();
        std::string fp2PathStr = "./TestFiles/" + models[i] + "/prob_estimates.txt";
        const char *fp2Path = fp2PathStr.c_str();
        std::cout<<"\n \n-----------------------------TEST FOR IMAGE"<<models[i]<<"-----------------------------\n";        
        printf("-----------------------------Probability Test-----------------------------\n");
        FILE *fp1 = fopen(fp1Path, "r"); 
        FILE *fp2 = fopen(fp2Path, "r"); 
    
        if (fp1 == NULL || fp2 == NULL) 
        { 
        printf("Error : Files not open"); 
        exit(0); 
        } 
    
        compareFiles(fp1, fp2, true); 
        fclose(fp1); 
        fclose(fp2); 

        printf("-------------------------------Labels Test-------------------------------\n");

        
        std::string fp3PathStr = "./TestFiles/" + models[i] + "/implementationLabels.txt";
        const char *fp3Path = fp3PathStr.c_str();
        std::string fp4PathStr = "./TestFiles/" + models[i] + "/labels_obtained.txt";
        const char *fp4Path = fp4PathStr.c_str();
        FILE *fp3 = fopen(fp3Path, "r"); 
        FILE *fp4 = fopen(fp4Path, "r"); 

        if (fp3 == NULL || fp4 == NULL) 
        {
        printf("Error : Files not open"); 
        exit(0); 
        } 
    
        compareFiles(fp3, fp4, true); 
    
        // closing both file 
        fclose(fp3); 
        fclose(fp4); 
        
    }

    printf("\n \n-------------------------------Global Test Result-------------------------------\n");
    printf("Threshold Used : %f\n", errorThreshold);
    printf("Total Errors : %d\n", globalErrors);
}

void singleTest(std::string modelName){    
        std::string fp1PathStr = "./TestFiles/" + modelName + "/implementationProb.txt";
        const char *fp1Path = fp1PathStr.c_str();
        std::string fp2PathStr = "./TestFiles/" + modelName + "/prob_estimates.txt";
        const char *fp2Path = fp2PathStr.c_str();
        std::cout<<"\n \n-----------------------------TEST FOR IMAGE"<<modelName<<"-----------------------------\n";        
        printf("-----------------------------Probability Test-----------------------------\n");
        FILE *fp1 = fopen(fp1Path, "r"); 
        FILE *fp2 = fopen(fp2Path, "r"); 
    
        if (fp1 == NULL || fp2 == NULL) 
        { 
        printf("Error : Files not open"); 
        exit(0); 
        } 
    
        compareFiles(fp1, fp2, true); 
        fclose(fp1); 
        fclose(fp2); 

        printf("-------------------------------Labels Test-------------------------------\n");

        
        std::string fp3PathStr = "./TestFiles/" + modelName + "/implementationLabels.txt";
        const char *fp3Path = fp3PathStr.c_str();
        std::string fp4PathStr = "./TestFiles/" + modelName + "/labels_obtained.txt";
        const char *fp4Path = fp4PathStr.c_str();
        FILE *fp3 = fopen(fp3Path, "r"); 
        FILE *fp4 = fopen(fp4Path, "r"); 

        if (fp3 == NULL || fp4 == NULL) 
        {
        printf("Error : Files not open"); 
        exit(0); 
        } 
    
        compareFiles(fp3, fp4, true); 
    
        // closing both file 
        fclose(fp3); 
        fclose(fp4); 
}
  
// Driver code 
int main() 
{ 
    globalTest();
    //singleTest("Op20C1");
    return 0; 
    
    
} 
