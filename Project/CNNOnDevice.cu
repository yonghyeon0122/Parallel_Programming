/*
	CNNonDevice.cu
	CNN conputation on device
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <algorithm> // std::max_element

#include "matrix.h"
#include "CNN.h"
#include "CNNOnDevice.h"
/*-------------------------------------------------
	Declaration
---------------------------------------------------*/
extern "C"
Matrix AllocateMatrix(int height, int width, int channel, int num_matrix);
Matrix AllocateDeviceMatrix(int height, int width, int channel, int num_matrix);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost, int elem_per_seg, int seg_idx);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice, int elem_per_seg, int seg_idx);
void FreeDeviceMatrix(Matrix* M);
void WriteFile(Matrix M, char* file_name);
void WriteFileRange(Matrix M, char* file_name, unsigned int start, unsigned int end);
Matrix ReshapeDeviceMatrix(Matrix M, int height, int width, int channel, int num_matrix);
float computeAccuracy(Matrix M, Matrix Ref, unsigned int num_element);

Matrix Convolution_ReLU(Matrix input);
Matrix MaxPool(Matrix input);
Matrix Affine1_ReLU(Matrix input, Matrix W);
Matrix Affine2(Matrix input, Matrix W);
/*--------------------------------------------------
	Kernels and biases are defined as const mem
----------------------------------------------------*/
extern __constant__ float W1_d[W1_WIDTH*W1_HEIGHT*W1_CHANNEL*W1_NUM_MATRIX]; // Kernel
extern __constant__ float b1_d[B1_WIDTH*B1_HEIGHT*B1_CHANNEL*B1_NUM_MATRIX]; // Bias
extern __constant__ float b2_d[B2_WIDTH*B2_HEIGHT*B2_CHANNEL*B2_NUM_MATRIX]; // Bias
extern __constant__ float b3_d[B3_WIDTH*B3_HEIGHT*B3_CHANNEL*B3_NUM_MATRIX]; // Bias

/*--------------------------------------------------
	Functions for computing on device
----------------------------------------------------*/
void CNNOnDevice(InOut inout_h, Params params_h)
{
    // Define parameters
    InOut inout_d;
    Params params_d;

    Matrix conv_out1, max_pool_out1, relu_out1, affine_out1, relu_out2, affine_out2;
    unsigned int result;
    float accuracy = 0.0f;
    //unsigned int num_inputs = INPUT_NUM_PER_COMPUTE;

    // Initialize device (GPU) matrixes
    // The matrix dimension is exactly same as the host matrix
    printf("Cuda Mem Allocation Starts\n");
    //inout_d.image = AllocateDeviceMatrix(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, INPUT_NUM_MATRIX);
    //inout_d.label = AllocateDeviceMatrix(LABEL_HEIGHT, LABEL_WIDTH, LABEL_CHANNEL, LABEL_NUM_MATRIX);
    params_d.W2 = AllocateDeviceMatrix(W2_HEIGHT, W2_WIDTH, W2_CHANNEL, W2_NUM_MATRIX);
    params_d.W3 = AllocateDeviceMatrix(W3_HEIGHT, W3_WIDTH, W3_CHANNEL, W3_NUM_MATRIX);
    printf("Cuda Mem Allocation Ends\n");

    // Copy Host matrixes to the Device ones
    // Question: Can we overlap the memory copy with the computation?
    CopyToDeviceMatrix(params_d.W2, params_h.W2, params_h.W2.width*params_h.W2.height, 0);
    CopyToDeviceMatrix(params_d.W3, params_h.W3, params_h.W3.width*params_h.W3.height, 0);

    // Kernels and biases are stored into constant memory
    printf("Constant Memory Allocation Starts\n");
    cudaMemcpyToSymbol(W1_d, params_h.W1.elements, W1_WIDTH*W1_HEIGHT*W1_CHANNEL*W1_NUM_MATRIX*sizeof(float));
    cudaMemcpyToSymbol(b1_d, params_h.b1.elements, B1_WIDTH*B1_HEIGHT*B1_CHANNEL*B1_NUM_MATRIX*sizeof(float));
    cudaMemcpyToSymbol(b2_d, params_h.b2.elements, B2_WIDTH*B2_HEIGHT*B2_CHANNEL*B2_NUM_MATRIX*sizeof(float));
    cudaMemcpyToSymbol(b3_d, params_h.b3.elements, B3_WIDTH*B3_HEIGHT*B3_CHANNEL*B3_NUM_MATRIX*sizeof(float));
    printf("Constant Memory Allocation Ends\n");
    
    // Start passing the input to the CNN layers
    // 1. Data transfer of part of the 10000 images
    //     Strategy: Allocate memory only for the part of the images (INPUT_NUM_PER_COMPUTE)
    //     Perform data transfer only for that batch of images
    //     The elements want to be processed: input_elements_per_compute
    //     Index of the segment: seg_idx
    unsigned int input_elements = INPUT_HEIGHT*INPUT_WIDTH*INPUT_CHANNEL*INPUT_NUM_PER_COMPUTE ;
    unsigned int seg_idx = 0;
    printf("Image Allocation: %d Images\n", INPUT_NUM_PER_COMPUTE );
    inout_d.image = AllocateDeviceMatrix(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, INPUT_NUM_PER_COMPUTE);    
    printf("Image Copy to Device Matrix\n");
    CopyToDeviceMatrix(inout_d.image, inout_h.image, input_elements, seg_idx);
    // 2. Perform convolution on the input segment
    //printf("Convolution ReLU \n");
    relu_out1 = Convolution_ReLU(inout_d.image);
    cudaDeviceSynchronize();
    //printf("Convolution ReLU done \n");

    //printf("Max pool \n");
    max_pool_out1 = MaxPool(relu_out1);
    cudaDeviceSynchronize();
    //printf("Max pool done\n");

    //printf("Affine1 ReLU \n");
    // Note: The matrix are reshaped 
    // e.g.) 12 x 12 x 1 x 300000 -> 1 x 4320 x 1 x 10000      
    int max_pool_out1_size = max_pool_out1.height * max_pool_out1.width * max_pool_out1.channel * W1_NUM_MATRIX;
    int max_pool_out1_num_matrix = max_pool_out1.num_matrix / W1_NUM_MATRIX;
    Matrix max_pool_out1_reshape =  ReshapeDeviceMatrix(max_pool_out1, 1, max_pool_out1_size, 1, max_pool_out1_num_matrix);
    
    relu_out2 = Affine1_ReLU(max_pool_out1_reshape, params_d.W2);
    cudaDeviceSynchronize();
    //printf("Affine1 ReLU done\n");  

    //printf("Affine2 \n");
    affine_out2 = Affine2(relu_out2, params_d.W3);
    cudaDeviceSynchronize();
    //printf("Affine2 done \n");


    // Functionality test
    /*
    printf("Functionality Test \n");
    Matrix relu_out1_h = AllocateMatrix(relu_out1.height, relu_out1.width, relu_out1.channel, relu_out1.num_matrix);
    CopyFromDeviceMatrix(relu_out1_h, relu_out1, relu_out1.height*relu_out1.width*
                                  relu_out1.channel*relu_out1.num_matrix, 0);
    char relu_out1_string[100] = "test_GPU/relu_out1_d_test.txt";
    WriteFile(relu_out1_h, relu_out1_string); 

    Matrix max_pool_out1_h = AllocateMatrix(max_pool_out1_reshape.height, max_pool_out1_reshape.width, max_pool_out1_reshape.channel, max_pool_out1_reshape.num_matrix);
    CopyFromDeviceMatrix(max_pool_out1_h, max_pool_out1_reshape, max_pool_out1.height*max_pool_out1.width*
                                  max_pool_out1.channel*max_pool_out1.num_matrix, 0);
    char max_pool_out1_string[100] = "test_GPU/max_pool_out1_d_test.txt";
    WriteFile(max_pool_out1_h, max_pool_out1_string); 
    
    Matrix relu_out2_h = AllocateMatrix(relu_out2.height, relu_out2.width, relu_out2.channel, relu_out2.num_matrix);
    CopyFromDeviceMatrix(relu_out2_h, relu_out2, relu_out2.height*relu_out2.width*
                                  relu_out2.channel*relu_out2.num_matrix, 0);
    char relu_out2_string[100] = "test_GPU/relu_out2_d_test.txt";
    WriteFile(relu_out2_h, relu_out2_string);
     
    Matrix w2_test = AllocateMatrix(params_d.W2.height, params_d.W2.width, params_d.W2.channel, params_d.W2.num_matrix);
    CopyFromDeviceMatrix(w2_test, params_d.W2, params_d.W2.height*params_d.W2.width*
                                  params_d.W2.channel*params_d.W2.num_matrix, 0);

    char w2_out_string[100] = "test_GPU/w2_out.txt";
    WriteFile(w2_test, w2_out_string);
    

    Matrix affine_out2_h = AllocateMatrix(affine_out2.height, affine_out2.width, affine_out2.channel, affine_out2.num_matrix);
    CopyFromDeviceMatrix(affine_out2_h, affine_out2, affine_out2.height*affine_out2.width*
                                  affine_out2.channel*affine_out2.num_matrix, 0);
    char affine_out2_string[100] = "test_GPU/affine_out2_d_test.txt";
    WriteFile(affine_out2_h, affine_out2_string);
    */    

    // Final classification by getting the max value
    Matrix affine_out2_h = AllocateMatrix(affine_out2.height, affine_out2.width,
                                             affine_out2.channel, affine_out2.num_matrix);
    CopyFromDeviceMatrix(affine_out2_h, affine_out2, affine_out2.height*affine_out2.width*
                                  affine_out2.channel*affine_out2.num_matrix, 0);

   
    for (int num_image = 0; num_image < INPUT_NUM_PER_COMPUTE; num_image++){
        float* startAddr = affine_out2_h.elements + 10*num_image;
        result = std::distance(startAddr, std::max_element(startAddr, startAddr+10));

        //printf("Result: %d, Label: %d\n", result, (int)inout_h.label.elements[num_image]);
        inout_h.output_d.elements[num_image] = result;

    }

    accuracy = computeAccuracy(inout_h.output_d, inout_h.label, INPUT_NUM_PER_COMPUTE);
    printf("Accuracy: %0.2f Percent \n", accuracy * 100.0);
    printf("No. of Input Images: %d\n", INPUT_NUM_PER_COMPUTE); 


    // Free Device matrices
    printf("Free Device Matrixex \n");
    FreeDeviceMatrix(&inout_d.image);
    FreeDeviceMatrix(&inout_d.label);
    FreeDeviceMatrix(&params_d.W2);
    FreeDeviceMatrix(&params_d.W3);
    FreeDeviceMatrix(&relu_out1);
    FreeDeviceMatrix(&max_pool_out1);
    FreeDeviceMatrix(&max_pool_out1_reshape);
    FreeDeviceMatrix(&affine_out2_h);

} // CNNOnDevice


Matrix AllocateDeviceMatrix(int height, int width, int channel, int num_matrix)
{
    Matrix Mdevice;
    int size = width * height * channel * num_matrix * sizeof(float);
    
    Mdevice.height = height;
    Mdevice.width = width;
    Mdevice.channel = channel;
    Mdevice.num_matrix = num_matrix;

    cudaMalloc((void**)&Mdevice.elements, size);
    //cudaMemset(Mdevice.elements, 0, size); // initialize to all zero
    return Mdevice;

}// AllocateDeviceMatrix


void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost, int elem_per_seg, int seg_idx)
{
    int elem_total = Mhost.width * Mhost.height * Mhost.channel * Mhost.num_matrix;
    //int size_total = elem_total * sizeof(float);
    int last_seg_idx = elem_total / elem_per_seg; // last segment index has to be specified for the accurate data transfer
    int last_elem_per_seg = elem_total % elem_per_seg; // remainder
    int size = elem_per_seg * sizeof(float); // as a default, size is defined based on elem_per_seg
    if (last_elem_per_seg) {last_seg_idx++;}

    int start = seg_idx * elem_per_seg;

    //Mdevice.height = Mhost.height;
    //Mdevice.width = Mhost.width;
    //Mdevice.channel = Mhost.channel;
    //Mdevice.num_matrix = Mhost.num_matrix;
   
    if (last_elem_per_seg!=0 && last_seg_idx==seg_idx) // If the elem_total is not a multiple of elem_per_seg
        size = last_elem_per_seg * sizeof(float);
        
    cudaMemcpy(Mdevice.elements, Mhost.elements+start, size, cudaMemcpyHostToDevice);

}// CopyToDeviceMatrix

void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice, int elem_per_seg, int seg_idx)
{
    int elem_total = Mhost.width * Mhost.height * Mhost.channel * Mhost.num_matrix;
    //int size_total = elem_total * sizeof(float);
    int last_seg_idx = elem_total / elem_per_seg; // last segment index has to be specified for the accurate data transfer
    int last_elem_per_seg = elem_total % elem_per_seg; // remainder
    int size = elem_per_seg * sizeof(float); // as a default, size is defined based on elem_per_seg
    if (last_elem_per_seg) {last_seg_idx++;}

    int start = seg_idx * elem_per_seg;   
 
    if (last_elem_per_seg!=0 && last_seg_idx==seg_idx) // If the elem_total is not a multiple of elem_per_seg
        size = last_elem_per_seg * sizeof(float);
       
    cudaMemcpy(Mhost.elements+start, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    

}// CopyFromDeviceMatrix

Matrix ReshapeDeviceMatrix(Matrix M, int height, int width, int channel, int num_matrix)
{
    Matrix Mdevice;
    int size = width * height * channel * num_matrix * sizeof(float);
    
    Mdevice.height = height;
    Mdevice.width = width;
    Mdevice.channel = channel;
    Mdevice.num_matrix = num_matrix;
    Mdevice.elements = M.elements;
    //cudaMalloc((void**)&Mdevice.elements, size);
    //cudaMemset(Mdevice.elements, 0, size); // initialize to all zero
    return Mdevice;


}// ReshapeDeviceMatrix


void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
} // FreeDeviceMatrix



