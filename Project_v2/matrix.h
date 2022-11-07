// matrix.h
// Header file
#ifndef _MATRIX_H_
#define _MATRIX_H_

// Structures
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int channel;
    unsigned int num_matrix;
    float* elements;
} Matrix;

typedef struct {
    Matrix W1, b1, W2, b2, W3, b3;
} Params;

typedef struct {
    Matrix image, label, output_d, output_h;
    
} InOut;

typedef struct {
    Matrix conv1_out, relu1_out, relu2_out, maxpool1_out, affine1_out, affine2_out;
} LayerOut;

#endif // _MATRIX_H_
