/*
	CNN header file
	CNN.h 
*/

#ifndef _CNN_H_
#define _CNN_H_

#define INPUT_NUM_PER_COMPUTE 10000

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_CHANNEL 1
#define INPUT_NUM_MATRIX 10000

#define LABEL_WIDTH 10000
#define LABEL_HEIGHT 1
#define LABEL_CHANNEL 1
#define LABEL_NUM_MATRIX 1

#define OUTPUT_WIDTH 10000
#define OUTPUT_HEIGHT 1
#define OUTPUT_CHANNEL 1
#define OUTPUT_NUM_MATRIX 1

#define W1_WIDTH 5
#define W1_HEIGHT 5
#define W1_CHANNEL 1
#define W1_NUM_MATRIX 30

#define B1_WIDTH 30
#define B1_HEIGHT 1
#define B1_CHANNEL 1
#define B1_NUM_MATRIX 1

#define W2_WIDTH 100
#define W2_HEIGHT 4320
#define W2_CHANNEL 1
#define W2_NUM_MATRIX 1

#define B2_WIDTH 100
#define B2_HEIGHT 1
#define B2_CHANNEL 1
#define B2_NUM_MATRIX 1 

#define W3_WIDTH 10
#define W3_HEIGHT 100
#define W3_CHANNEL 1
#define W3_NUM_MATRIX 1

#define B3_WIDTH 10
#define B3_HEIGHT 1
#define B3_CHANNEL 1
#define B3_NUM_MATRIX 1 

#endif // _CNN_H_

