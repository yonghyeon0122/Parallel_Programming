/*
	CNNOnHost.cpp
	Convolutional Neural Network on the Host Side
*/

#define MAX_POOL_SIZE 2

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <algorithm> // std::max_element

#include "matrix.h"
#include "CNN.h"
/*----------------------------------------------------------
	Function Declaration
-----------------------------------------------------------*/
extern "C"
Matrix AllocateMatrix(int height, int width, int channel, int num_matrix);
Matrix ConvolutionHost(Matrix input, unsigned int num_input, Matrix kernel, Matrix b);
void WriteFile(Matrix M, char* file_name);
Matrix MaxPool(Matrix input, unsigned int max_pool_size);
Matrix ReLU(Matrix input);
void ReshapeMatrix(Matrix M, int height, int width, int channel, int num_matrix);
Matrix Affine(Matrix input, Matrix W, Matrix b);
float computeAccuracy(Matrix M, Matrix Ref, unsigned int num_element);

/*----------------------------------------------------------
 	Starts
-----------------------------------------------------------*/
void CNNOnHost(InOut inout, Params params)
{
    Matrix conv_out1, max_pool_out1, relu_out1, affine_out1, relu_out2, affine_out2;
    unsigned int result;
    float accuracy = 0.0f;
    unsigned int num_inputs = INPUT_NUM_PER_COMPUTE;

    // Iterate the classification for each input image
    for(unsigned int num_image = 0; num_image < num_inputs; num_image++){
        // Convolution layer
        conv_out1 = ConvolutionHost(inout.image, num_image, params.W1, params.b1);
        relu_out1 = ReLU(conv_out1);
        max_pool_out1 = MaxPool(relu_out1, MAX_POOL_SIZE);
        // Fully connected layers
        ReshapeMatrix(max_pool_out1, 1, max_pool_out1.width*max_pool_out1.height*max_pool_out1.channel*max_pool_out1.num_matrix, 1, 1); 
        affine_out1 = Affine(max_pool_out1, params.W2, params.b2);
        relu_out2 = ReLU(affine_out1);
        affine_out2 = Affine(relu_out2, params.W3, params.b3);
        // Classification (get the index of the largest output)
        result = std::distance(affine_out2.elements, std::max_element(affine_out2.elements, affine_out2.elements + 10));        
        //printf("Result: %d, Label: %d\n", result, (int)inout.label.elements[num_image]);
        inout.output_h.elements[num_image] = result;
    }

    accuracy = computeAccuracy(inout.output_h, inout.label, num_inputs);
    printf("Accuracy: %0.2f Percent \n", accuracy * 100.0);
    printf("No. of Input Images: %d\n", num_inputs);
    
    // File write for testing
    /*
    char conv_out1_string[100] = "test/convout1_test.txt";
    WriteFile(conv_out1, conv_out1_string);  
    char max_pool_out1_string[100] = "test/maxpoolout1_test.txt";
    WriteFile(max_pool_out1, max_pool_out1_string); 
    char relu_out1_string[100] = "test/reluout1_test.txt";
    WriteFile(relu_out1, relu_out1_string);     
    char affine_out1_string[100] = "test/affineout1_test.txt";
    WriteFile(affine_out1, affine_out1_string);
    char relu_out2_string[100] = "test/reluout2_test.txt";
    WriteFile(relu_out2, relu_out2_string);
    char affine_out2_string[100] = "test/affineout2_test.txt";
    WriteFile(affine_out2, affine_out2_string);
    */

}// CNNOnHost


Matrix ConvolutionHost(Matrix input, unsigned int num_input, Matrix kernel, Matrix b)
{
    // Define parameters
    unsigned int out_width = input.width - kernel.width + 1;
    unsigned int out_height = input.height - kernel.height + 1;
    unsigned int stride = 1;
    unsigned int size_output = out_width * out_height;
    unsigned int size_image = input.width * input.height;
    unsigned int size_kernel = kernel.width * kernel.height;
    unsigned int w_image = input.width;
    unsigned int w_kernel = kernel.width;
    float sum = 0.0f;

    // Initialize the output matrix
    // Only the dimension of the matrix is shrinked
    Matrix output;
    output = AllocateMatrix(out_height, out_width, kernel.channel, kernel.num_matrix);

    // Get one of the image
    // num_element'th image will be grabbed from the input array
    // Then, perform Convolution
    for (int num_kernel = 0; num_kernel < kernel.num_matrix; num_kernel++) // Select 1 out of 30 kernels
    for (int row_o = 0; row_o < out_height; row_o = row_o + stride) // Output matrix indexing
    for (int col_o = 0; col_o < out_width; col_o = col_o + stride){
        sum = 0.0f;
        unsigned int start_idx = num_input*size_image; // start_idx depends on the image selection
        for (int row_k = 0; row_k < kernel.height; row_k++){ // Kernel indexing 
            for (int col_k = 0; col_k < kernel.width; col_k++){
               sum += input.elements[start_idx + w_image*(row_o+row_k) + (col_o+col_k)] *
                             kernel.elements[num_kernel*size_kernel + (w_kernel*row_k + col_k)];

            //printf("num_kernel: %d, row_o: %d, col_o: %d, row_k: %d, col_k: %d\n", num_kernel, row_o, col_o, row_k, col_k);
            }
        }
        // Assign the convolution result to the output matrix
        // Also, bias is added
        output.elements[num_kernel*size_output + (out_width*row_o + col_o)] = sum + b.elements[num_kernel];    
 
    }

    return output;
}// ConvolutionHost

Matrix MaxPool(Matrix input, unsigned int max_pool_size)
{
    //Define parameters
    unsigned int out_width = input.width / max_pool_size;
    unsigned int out_height = input.height / max_pool_size;
    unsigned int size_input = input.width * input.height;
    unsigned int size_output = out_width * out_height;
    float* local_array = (float*) malloc(max_pool_size*max_pool_size*sizeof(float));
    //float local_array[MAX_POOL_SIZE * MAX_POOL_SIZE]; 

    // Initialize the output matrix
    // After the maxpooling, the dimension of the matrix is shrinked
    Matrix output;
    output = AllocateMatrix(out_height, out_width, input.channel, input.num_matrix);

    // Get on of the image
    // num_element'th image will be grabbed from the input array
    // Then, perform maxpooling
    for (int num_matrix = 0; num_matrix < input.num_matrix; num_matrix++) // Select 1 out of 30 matrixes
    for (int row = 0; row < input.height; row = row + max_pool_size)
    for (int col = 0; col < input.width; col = col + max_pool_size){
       unsigned int row_o = row/max_pool_size;
       unsigned int col_o = col/max_pool_size;
       unsigned int start_idx = num_matrix*size_input; // Select 1 out of 30 matrixes
       float max = 0.0f;
       for (int row_l = 0; row_l < max_pool_size; row_l++)
           for (int col_l = 0; col_l < max_pool_size; col_l++){
               local_array[col_l + row_l*max_pool_size]
                       = input.elements[start_idx + input.width*(row+row_l) + (col+col_l)];      
      
           }
       max = *std::max_element(local_array, local_array+4);
       //printf("1st: %f, 2nd: %f, 3rd: %f, 4th: %f, max: %f\n", local_array[0], local_array[1], local_array[2], local_array[3], max);
       output.elements[num_matrix*size_output + (out_width*row_o + col_o)] = max;
    }
    

       
    return output;

} // MaxPool

Matrix ReLU(Matrix input)
{
    // Define parameters
    unsigned int num_total_elements = input.height * input.width * input.channel * input.num_matrix;

    // Initialize the output matrix (to all zero)
    // Relu doesn't change the dimension
    Matrix output;
    output = AllocateMatrix(input.height, input.width, input.channel, input.num_matrix);

    // Rectify the elements
    // Negative elements become zero
    for (unsigned int n = 0; n < num_total_elements; n++){
        if(input.elements[n] > 0)
            output.elements[n] = input.elements[n];
    }
    return output;
}// ReLU

Matrix Affine(Matrix input, Matrix W, Matrix b)
{
    // Define parameters
    //int num_total_element_input = input.width * input.height * input.channel * input.num_matrix;

    // Initialize output matrix
    // The output dimension of the fully connected network is the height of the weight matrix
    Matrix output;
    output = AllocateMatrix(1, W.width, 1, 1);

    // Matrix multiplication
    // Indexing is based on row and col of the W matrix
    for (int col = 0; col < W.width; col++){
        float sum = 0.0f;
        for (int row = 0; row < W.height; row++){
            sum += input.elements[row] * W.elements[row*W.width + col];
        }
        output.elements[col] = sum + b.elements[col];// Add bias
    }
            
    return output;   
    
} // Affine

float computeAccuracy(Matrix M, Matrix Ref, unsigned int num_elements)
{
    float accuracy = 0.0f;
    float matchCount = 0.0f;
    //unsigned int num_elements = M.width * M.height * M.channel * M.num_matrix;    

    for(int n = 0; n < num_elements; n++){
        if((int)M.elements[n] == (int)Ref.elements[n])
            matchCount = matchCount + 1.0;
    } 

    accuracy = matchCount / (float)num_elements;

    return accuracy;
}// accuracy
