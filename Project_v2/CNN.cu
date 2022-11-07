/*
	Simple Convolutional Neural Network
	
	Input: MNIST dataset
	Output: Classified result

	Layer order: Conv - ReLU - MaxPooling - Affine - ReLU - Affine - Softmax

*/


/* Includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "CNN.h"
#include "CNNOnDevice.h"
#include "util.h"
/*------------------------------------------------------
	Declarations, Forward
-------------------------------------------------------*/

extern "C" 
//void CNNOnHost(InOut inout, Params params);
Matrix AllocateMatrix(int height, int width, int channel, int num_matrix);
void FreeMatrix(Matrix* M);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void CNNOnHost(InOut inout, Params params);
void CNNOnDevice(InOut inout, Params params);

/*------------------------------------------------------
    Main Starts
	
    Arguments:
    Inputs
        argv[1] = image_input
	argv[2] = label_input
    Convolution layer
	argv[3] = W1 (kernel)
	argv[4] = b1
    Affine layers
	argv[5] = W2
	argv[6] = b2
	argv[7] = W3
	argv[8]	= b3
-------------------------------------------------------*/

int main(int argc, char** argv){

    // Device Property Check
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("The number of device is: %d\n", dev_count);

    cudaDeviceProp dev_prop;
    for (int i=0; i<dev_count;i++){
        cudaGetDeviceProperties(&dev_prop,i);
        printf("Total global memory (byte): %zu\n", dev_prop.totalGlobalMem);
        printf("Shared memory per block (byte): %zu\n", dev_prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", dev_prop.regsPerBlock);
        printf("Max threads per block: %d\n", dev_prop.maxThreadsPerBlock);
        //printf("Max threads per SM: %d\n", dev_prop.maxThreadsPerSM);
        //printf("Max thread blocks per SM: %d\n", dev_prop.maxThreadBlocksPerSM);
        printf("Max grid size x: %d\n", dev_prop.maxGridSize[0]);
        printf("Max grid size y: %d\n", dev_prop.maxGridSize[1]);
        printf("Max grid size z: %d\n", dev_prop.maxGridSize[2]);
        printf("Total constant memory (byte): %zu\n", dev_prop.totalConstMem);
        printf("Multi processor count: %d\n", dev_prop.multiProcessorCount);
        printf("Warp size: %d\n", dev_prop.warpSize);
    }


    /*--------------------------------------
	Code Starts
    ----------------------------------------*/

    InOut inout;
    Params params;
    LayerOut layerout; // Debugging purpose
	
    // Decide the matrix dimension and initialize into all zero
    printf("Paramter Initialization Starts\n");
    inout.image = AllocateMatrix(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, INPUT_NUM_MATRIX);   
    inout.label = AllocateMatrix(LABEL_HEIGHT, LABEL_WIDTH, LABEL_CHANNEL, LABEL_NUM_MATRIX);
    inout.output_h = AllocateMatrix(OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL, OUTPUT_NUM_MATRIX);
    inout.output_d = AllocateMatrix(OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL, OUTPUT_NUM_MATRIX);
    params.W1 = AllocateMatrix(W1_HEIGHT, W1_WIDTH, W1_CHANNEL, W1_NUM_MATRIX);
    params.b1 = AllocateMatrix(B1_HEIGHT, B1_WIDTH, B1_CHANNEL, B1_NUM_MATRIX);
    params.W2 = AllocateMatrix(W2_HEIGHT, W2_WIDTH, W2_CHANNEL, W2_NUM_MATRIX);
    params.b2 = AllocateMatrix(B2_HEIGHT, B2_WIDTH, B2_CHANNEL, B2_NUM_MATRIX);
    params.W3 = AllocateMatrix(W3_HEIGHT, W3_WIDTH, W3_CHANNEL, W3_NUM_MATRIX);
    params.b3 = AllocateMatrix(B3_HEIGHT, B3_WIDTH, B3_CHANNEL, B3_NUM_MATRIX);
    printf("Parameter Initialization Done \n");

    //// Read the file and allocate the data
    printf("Paramter Allocation Starts\n");
    unsigned sizeImage = ReadFile(&inout.image, argv[1]);
    unsigned sizeLabel = ReadFile(&inout.label, argv[2]);
    unsigned sizeW1 = ReadFile(&params.W1, argv[3]);
    unsigned sizeb1 = ReadFile(&params.b1, argv[4]);
    unsigned sizeW2 = ReadFile(&params.W2, argv[5]);
    unsigned sizeb2 = ReadFile(&params.b2, argv[6]);
    unsigned sizeW3 = ReadFile(&params.W3, argv[7]);
    unsigned sizeb3 = ReadFile(&params.b3, argv[8]);
    printf("Paramter Allocation Done\n");


    //// Write the results into file (Just for testing)
    /*
    char image_output[100] = "test/image_output_test.txt";
    char label_output[100] = "test/label_output_test.txt";
    char W1_output[100] = "test/W1_output_test.txt";
    char b1_output[100] = "test/b1_output_test.txt";
    char W2_output[100] = "test/W2_output_test.txt";
    char b2_output[100] = "test/b2_output_test.txt";
    char W3_output[100] = "test/W3_output_test.txt";
    char b3_output[100] = "test/b3_output_test.txt";
   
    WriteFile(inout.image, image_output);
    WriteFile(inout.label, label_output);
    WriteFile(params.W1, W1_output);
    WriteFile(params.b1, b1_output);
    WriteFile(params.W2, W2_output);
    WriteFile(params.b2, b2_output);
    WriteFile(params.W3, W3_output);
    WriteFile(params.b3, b3_output);
    */

    /*-----------------------------------------------------
	CNN on the host (CPU) side
    --------------------------------------------------------*/
    printf("\nCPU CNN Classification Starts\n");
    CNNOnHost(inout, params);
    //TIME_IT("CPU CNN", 1, CNNOnHost(inout, params););
    printf("\nCPU CNN Classification Done\n");

    /*-----------------------------------------------------
        CNN on the device (GPU) side
    --------------------------------------------------------*/
    printf("\nGPU CNN Classification Starts\n");
    CNNOnDevice(inout, params);
    //TIME_IT("GPU CNN", 1,CNNOnDevice(inout, params););
    printf("\nGPU CNN Classification Done\n");
  
    //// FreeMatrix
    printf("Free Matrixes\n");
    FreeMatrix(&inout.image);
    FreeMatrix(&inout.label);
    FreeMatrix(&params.W1);
    FreeMatrix(&params.b1);
    FreeMatrix(&params.W2);
    FreeMatrix(&params.b2);
    FreeMatrix(&params.W3);
    FreeMatrix(&params.b3);
    FreeMatrix(&inout.output_d);
    FreeMatrix(&inout.output_h);


}// main
