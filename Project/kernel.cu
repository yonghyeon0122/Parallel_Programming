/*
	Kernels for CNN
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <algorithm> // std::max_element

#include "matrix.h"
#include "CNNOnDevice.h"
#include "CNN.h"

/*-------------------------------------------------
        Declaration
---------------------------------------------------*/
Matrix AllocateDeviceMatrix(int height, int width, int channel, int num_matrix);
Matrix Convolution_ReLU(Matrix input);
Matrix MaxPool(Matrix input);
Matrix Affine1_ReLU(Matrix input, Matrix W);
Matrix Affine2(Matrix input, Matrix W);

__global__ void Convolution_ReLU_Kernel(Matrix input, Matrix output);
__global__ void MaxPool_Kernel(Matrix input, Matrix output);
__global__ void Affine1_ReLU_Kernel(Matrix input, Matrix W, Matrix output);
__global__ void Affine2_Kernel(Matrix input, Matrix W, Matrix output);

/*--------------------------------------------------
        Kernels and biases are defined as const mem
----------------------------------------------------*/
__constant__ float W1_d[W1_WIDTH*W1_HEIGHT*W1_CHANNEL*W1_NUM_MATRIX]; // Kernel
__constant__ float b1_d[B1_WIDTH*B1_HEIGHT*B1_CHANNEL*B1_NUM_MATRIX]; // Bias
__constant__ float b2_d[B2_WIDTH*B2_HEIGHT*B2_CHANNEL*B2_NUM_MATRIX]; // Bias
__constant__ float b3_d[B3_WIDTH*B3_HEIGHT*B3_CHANNEL*B3_NUM_MATRIX]; // Bias

// Convolution and the ReLU activation
// The input of this function is a "set" of input images rather than a single image 
Matrix Convolution_ReLU(Matrix input){

    // Define parameters
    unsigned int out_width = INPUT_WIDTH - W1_WIDTH + 1;
    unsigned int out_height = INPUT_HEIGHT - W1_HEIGHT + 1;
    unsigned int size_output = out_width * out_height;
    unsigned int size_image = INPUT_WIDTH * INPUT_HEIGHT;
    unsigned int size_kernel = W1_WIDTH * W1_HEIGHT;
    float sum = 0.0f;

    // Initialize the output matrix
    // The set of output matrix depends on the 
    Matrix output;
    output = AllocateDeviceMatrix(out_height, out_width, W1_CHANNEL, INPUT_NUM_PER_COMPUTE * W1_NUM_MATRIX);

    // Set block and grid dimension
    // Each image is allocated to each thread block
    // Output will be the 30 copies of convoluted output array
    // The kernel used by each thread block depends on the grid dimension
    dim3 blockDim(input.width, input.height, 1);
    dim3 gridDim(input.num_matrix * W1_NUM_MATRIX, 1, 1); 

    // Launch kernel
    Convolution_ReLU_Kernel<<<gridDim, blockDim>>>(input, output);
 
    return output;
   
} // Convolution_ReLU

Matrix MaxPool(Matrix input)
{
    // Define parameters
    unsigned int output_width = input.width / MAX_POOL_SIZE;
    unsigned int output_height = input.height / MAX_POOL_SIZE;
    unsigned int size_input = input.width * input.height;
    unsigned int size_output = output_width * output_height;
 
    // Initialize the output matrix
    // After the maxpooling, the dimension of the matrix is shrinked
    Matrix output;
    output = AllocateDeviceMatrix(output_height, output_width, input.channel, input.num_matrix);

    // Set block and grid dimension
    // Output has the same number of matrixes as the input
    // Each matrix requires out_width x out_height threads
    // e.g.) 12 x 12 threads are required
    // Using maximum number of threads per block gives better performance
    // We can fit seven matrixes per block, which gives 7 x 12 x 12 = 1008 threads per block  
    unsigned int num_matrix_per_block = 7;
    unsigned int grid_x = input.num_matrix / num_matrix_per_block;
    if(input.num_matrix % num_matrix_per_block){grid_x++;}

    dim3 blockDim(output_width, output_height, num_matrix_per_block);
    dim3 gridDim(grid_x, 1, 1);

    // Launch kernel
    MaxPool_Kernel<<<gridDim, blockDim>>>(input, output);
    
    return output;
} // MaxPool


Matrix Affine1_ReLU(Matrix input, Matrix W)
{
    // Perform Affine1 and ReLU operation on the batch of the input matrixes
    // e.g.) If the INPUT_NUM_PER_COMPUTATION is 10000, 10000 Affine outputs will be generated

    // Define parameters
    
    // Initialize the output matrix
    Matrix output;
    output = AllocateDeviceMatrix(1, W.width, 1, INPUT_NUM_PER_COMPUTE);

    // Set block and grid dimension
    // This matrix multiplication is a (1 x m) * (m x n) computation
    // The input is a 1 dimension matrix (with large number of elements)
    int block_x = BLOCK_DIM_AFFINE1;
    int grid_x = INPUT_NUM_PER_COMPUTE / BLOCK_DIM_AFFINE1;
    if (INPUT_NUM_PER_COMPUTE % BLOCK_DIM_AFFINE1) {grid_x ++;}
    dim3 blockDim(block_x, 1, 1);
    dim3 gridDim(grid_x, 1, 1);
    
    // Define partial sum array
    //float* partialSumArray;
    //unsigned int row_size_p = grid_x; unsigned int col_size_p = output.width;
    //unsigned int size = row_size_p * col_size_p * sizeof(float) ; 
    //cudaMalloc((void**)&partialSumArray, size);

    // Launch Kernel
    Affine1_ReLU_Kernel<<<gridDim, blockDim>>>(input, W, output);
    
    return output;
}// Affine1_ReLU

Matrix Affine2(Matrix input, Matrix W)
{
    // Perform Affine2  operation on the batch of the input matrixes
    // e.g.) If the INPUT_NUM_PER_COMPUTATION is 10000, 10000 Affine outputs will be generated

    // Define parameters

    // Initialize the output matrix
    Matrix output;
    output = AllocateDeviceMatrix(1, W.width, 1, INPUT_NUM_PER_COMPUTE);

    // Set block and grid dimension
    // This matrix multiplication is a (1 x m) * (m x n) computation
    // The input is a 1 dimension matrix (with large number of elements)
    int block_x = BLOCK_DIM_AFFINE2;
    int grid_x = INPUT_NUM_PER_COMPUTE / BLOCK_DIM_AFFINE2;
    if (INPUT_NUM_PER_COMPUTE % BLOCK_DIM_AFFINE2) {grid_x ++;}
    dim3 blockDim(block_x, 1, 1);
    dim3 gridDim(grid_x, 1, 1);

    // Define partial sum array
    //float* partialSumArray;
    //unsigned int row_size_p = grid_x; unsigned int col_size_p = output.width;
    //unsigned int size = row_size_p * col_size_p * sizeof(float) ; 
    //cudaMalloc((void**)&partialSumArray, size);

    // Launch Kernel
    Affine2_Kernel<<<gridDim, blockDim>>>(input, W, output);

    return output;
}// Affine2


__global__ void Convolution_ReLU_Kernel(Matrix input, Matrix output)
{
    // Each input image is stored into shared memory
    __shared__ float image[INPUT_HEIGHT][INPUT_WIDTH];

    // Indexing
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;
    int image_idx = bx / W1_NUM_MATRIX ; 
    int image_size = INPUT_WIDTH*INPUT_HEIGHT ;
    int kernel_idx = bx % W1_NUM_MATRIX ;
    int kernel_size = W1_WIDTH*W1_HEIGHT ;
    int output_size = output.width * output.height;
    int output_width = output.width;
    int output_height = output.height;
    int stride = 1;
    
    // Load to the shared memory
    // Each treads select data from the input and load it to the shared memory
    // The image selection is decided by the block index
    // We are using 30 blocks to compute one image since we have 30 kernels
    image[ty][tx] = input.elements[image_idx * image_size + ty*INPUT_WIDTH + tx];

    // Thread synchronization is necessary
    __syncthreads();  
    
    // Convolution starts
    // W1_d and b1_d arrays are already defined globally
    // 1. The kernel index (kernel_idx) is defind by the block index (remainder)
    // 2. The striding (selection of the output matrix index) is decided by the thread indexes (tx, ty)
    //   e.g.) tx and ty ranges from 0 to out_width and out_height respectively
    // 3. The only for loop required is a computation of a single output element (i.e., one kernel computation)
    // 4. Finally, addthe bias and pass to the ReLU activation
    float sum = 0.0f;
    if(tx < output_width && ty < output_height){
    for (int row_k = 0; row_k < W1_HEIGHT; row_k = row_k + stride)
        for (int col_k = 0; col_k < W1_WIDTH; col_k = col_k + stride){
            sum += image[ty+row_k][tx+col_k] * W1_d[kernel_idx * kernel_size + row_k*W1_WIDTH + col_k];
        }
    sum = sum + b1_d[kernel_idx];
    // ReLU
    if (sum < 0)
        sum = 0.0f;

    output.elements[bx * output_size + ty * output_width + tx] = sum;
    }
}// ConvolutionKernel

__global__ void MaxPool_Kernel(Matrix input, Matrix output)
{
    // Shared memory might not be necessary since the data is not used repeatedly

    // Local array
    float M_local[MAX_POOL_SIZE*MAX_POOL_SIZE];
 
    // Indexing   
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int bx = blockIdx.x;
    int stride = MAX_POOL_SIZE;
    int start_output = bx * blockDim.x * blockDim.y * blockDim.z;
    int start_input = start_output * MAX_POOL_SIZE * MAX_POOL_SIZE;
    int input_size = input.width*input.height;
    int output_size = output.width*output.height;
    float value;

    // Maxpooling Starts
    // The computation should be done within the number of input matrixes
    // 1. Each thread selects MAX_POOL_SIZE x MAX_POOL_SIZE elements from the input matrixes
    // 2. Get the maximum one from the local array (M_local)
    // 3. Assign the maximum one into the output array
    float max = 0.0f;
    if((tz + blockDim.z * bx) < input.num_matrix){
    for (int row_l = 0; row_l < MAX_POOL_SIZE; row_l++)
        for (int col_l = 0; col_l < MAX_POOL_SIZE; col_l++){
            value  = input.elements[start_input + tz * input_size + (ty*stride+row_l)*input.width+(tx*stride+col_l)];
            M_local[row_l*MAX_POOL_SIZE + col_l] = value;
         }
    
    max = M_local[0];
    for (int row_l = 0; row_l < MAX_POOL_SIZE; row_l++)
         for (int col_l = 0; col_l < MAX_POOL_SIZE; col_l++){
              value = M_local[row_l*MAX_POOL_SIZE+col_l];
              if(value > max)
                   max = value;
         }      
    output.elements[start_output + tz * output_size + ty * output.width + tx] = max;       

    }//if    

} // MaxPool_Kernel

__global__ void Affine1_ReLU_Kernel(Matrix input, Matrix W, Matrix output)
{
    // Shared memory
    __shared__ float X_p[BLOCK_DIM_AFFINE1];
    __shared__ float W_p[W2_WIDTH];
    __shared__ float output_s[BLOCK_DIM_AFFINE1][W2_WIDTH];
    
    // Indexing
    int tx = threadIdx.x; 
    int bx = blockIdx.x;
    int phase_total = W.height;
    int W_width = W.width;

    // The thread idx and the block dimension decides the image want to be processed
    int num_matrix = input.num_matrix; // The number of image being processed
    int num_element_i = input.height * input.width;
    int num_total_element_i = num_element_i * input.channel * input.num_matrix;
    int idx = tx + bx * blockDim.x; // meaning which image we are going to process
    int num_element_o = output.height * output.width;       

    // Initialize the output_s array
    // Alread all zero?
    for (int col_w = 0; col_w < W2_WIDTH; col_w++)
         output_s[tx][col_w] = 0.0f;

    __syncthreads();    


    // Start matrix multiplication
    float value = 0.0f;
    for (int phase_n = 0; phase_n < phase_total; phase_n++){
        // Load data to Xp
        int idx_input = num_element_i * idx + phase_n;
        if (idx_input < num_total_element_i)  
            X_p[tx] = input.elements[idx_input];
        else
            X_p[tx] = 0.0f;
        
        __syncthreads(); 
        // Load data to Wp
        // Warning, tx should be larger than W_width (e.g. 100, 10)
        if(tx < W_width){
            W_p[tx] = W.elements[tx + phase_n * W_width];      
            //W_p[tx] = 1.0f; 
        }

        __syncthreads();
    
        // The image index should not be larger than the total number of images (matrixes)
        if (idx < num_matrix){
            for (int col_w = 0; col_w < W_width; col_w++){
                //int idx_output = col_w + num_element_o * idx; // Ouput result index. Depends on col_w and the image index
                value = X_p[tx] * W_p[col_w];
                //value = X_p[tx];
                //value = W_p[col_w]; 
                //output.elements[idx_output] += value;
                output_s[tx][col_w] += value; // Accumulate the result to the shared memory, output_s
            }
        }
        
        __syncthreads();
    }

    // Add bias and ReLU
    if(idx < num_matrix){
        for (int col_w = 0; col_w < W.width; col_w++){
             int idx_output = col_w + num_element_o * idx;
             //value = output[idx_output];
            value = output_s[tx][col_w]; // Get the value from the shared memory, output_s
            value += b2_d[col_w]; // Add Bias2
            if(value < 0) // ReLU
                value = 0.0f;
            output.elements[idx_output] = value; // Assign the final result to the global memory, output
        }
    }
   
}// Affine_Kernel1

__global__ void Affine2_Kernel(Matrix input, Matrix W, Matrix output)
{
    // Shared memory
    __shared__ float X_p[BLOCK_DIM_AFFINE2];
    __shared__ float W_p[W3_WIDTH];
    __shared__ float output_s[BLOCK_DIM_AFFINE2][W3_WIDTH];

    // Indexing
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int phase_total = W.height;
    int W_width = W.width;


    // The thread idx and the block dimension decides the image want to be processed
    int num_matrix = input.num_matrix; // The number of image being processed
    int num_element_i = input.height * input.width;
    int num_total_element_i = num_element_i * input.channel * input.num_matrix;
    int idx = tx + bx * blockDim.x; // meaning which image we are going to process
    int num_element_o = output.height * output.width;

    // Initialize the output_s array
    // Alread all zero?
    for (int col_w = 0; col_w < W3_WIDTH; col_w++)
         output_s[tx][col_w] = 0.0f;

    __syncthreads();


    // Start matrix multiplication
    float value = 0.0f;
    for (int phase_n = 0; phase_n < phase_total; phase_n++){
        // Load data to Xp
        int idx_input = num_element_i * idx + phase_n;
        if (idx_input < num_total_element_i)
            X_p[tx] = input.elements[idx_input];
        else
            X_p[tx] = 0.0f;

        __syncthreads();
        // Load data to Wp
        // Warning, tx should be larger than W_width (e.g. 100, 10)
        if(tx < W_width){
            W_p[tx] = W.elements[tx + phase_n * W_width];
            //W_p[tx] = 1.0f; 
        }

        __syncthreads();


        // The image index should not be larger than the total number of images (matrixes)
        if (idx < num_matrix){
            for (int col_w = 0; col_w < W_width; col_w++){
                //int idx_output = col_w + num_element_o * idx; // Ouput result index. Depends on col_w and the image index
                value = X_p[tx] * W_p[col_w];
                //value = X_p[tx];
                //value = W_p[col_w]; 
                //output.elements[idx_output] += value;
                output_s[tx][col_w] += value; // Accumulate the result to the shared memory, output_s
            }
        }

        __syncthreads();
    }

    // Add bias 
    if(idx < num_matrix){
        for (int col_w = 0; col_w < W.width; col_w++){
             int idx_output = col_w + num_element_o * idx;
             //value = output[idx_output];
            value = output_s[tx][col_w]; // Get the value from the shared memory, output_s
            value += b3_d[col_w]; // Add Bias3
            //if(value < 0) // ReLU
            //    value = 0.0f;
            output.elements[idx_output] = value; // Assign the final result to the global memory, output
        }
    }

}// Affine2_Kernel

