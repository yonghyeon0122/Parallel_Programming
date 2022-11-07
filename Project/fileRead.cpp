/*
 	fileRead.cpp
	Helper functions:
	
	ReadFile()
	ReadParamsFile()
*/

/* Includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "matrix.h"

/*------------------------------------------------------
         Export C Interface
-------------------------------------------------------*/
extern "C"
Matrix AllocateMatrix(int height, int width, int channel, int num_matrix);
void FreeMatrix(Matrix* M);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void WriteFileRange(Matrix M, char* file_name, unsigned int start, unsigned int end);
void ReshapeMatrix(Matrix M, int height, int width, int channel, int num_matrix);

// Read a floating point matrix from file
// Could be an input image or weight & bias matrix
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int num_total_elements = M->width * M->height * M->num_matrix * M->channel;
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < num_total_elements; i++)
        fscanf(input, "%f", &(M->elements[i]));
    return num_total_elements;
}// ReadFile

void WriteFile(Matrix M, char* file_name)
{
    unsigned int size = M.width * M.height * M.channel * M.num_matrix ;
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < size; i++) {
        fprintf(output, "%f ", M.elements[i]);
    } 
} // WriteFile

void WriteFileRange(Matrix M, char* file_name, unsigned int start, unsigned int end)
{
    FILE* output = fopen(file_name, "w");
    for (unsigned i = start; i < end; i++) {
        fprintf(output, "%f ", M.elements[i]);
    }
} // WriteFile


Matrix AllocateMatrix(int height, int width, int channel, int num_matrix)
{
    Matrix M;
    M.width = width;
    M.height = height;
    M.channel = channel;
    M.num_matrix = num_matrix;
    M.elements = NULL;

    int size = M.width * M.height * M.channel * M.num_matrix;
    M.elements = (float*) malloc(size*sizeof(float));

    // Initialize elements to all zero
    for(unsigned int i = 0; i < size; i++)
	M.elements[i] = 0.0f;

    return M;
    

} // AllocateMatrix

void ReshapeMatrix(Matrix M, int height, int width, int channel, int num_matrix)
{
    M.height = height;
    M.width = width;
    M.channel = channel;
    M.num_matrix = num_matrix;
}// ReshapeMatrix

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
     free(M->elements);
     M->elements = NULL;
} // FreeMatrix

