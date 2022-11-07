# Make file
# To make the extern available for the constant memory definition, use -rdf=true flag option

NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" -rdc=true 
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = CNN
OBJ	        = CNN.o fileRead_cpp.o CNNOnHost_cpp.o CNNOnDevice_cu.o kernel_cu.o

default: $(EXE)

CNN.o: CNN.cu CNN.h matrix.h CNNOnDevice.h util.h 
	$(NVCC) -c -o $@ CNN.cu $(NVCC_FLAGS)

fileRead_cpp.o: fileRead.cpp matrix.h
	$(NVCC) -c -o $@ fileRead.cpp $(NVCC_FLAGS)

CNNOnHost_cpp.o: CNNOnHost.cpp matrix.h CNN.h
	$(NVCC) -c -o $@ CNNOnHost.cpp $(NVCC_FLAGS)

CNNOnDevice_cu.o: CNNOnDevice.cu matrix.h CNN.h CNNOnDevice.h
	$(NVCC) -c -o $@ CNNOnDevice.cu $(NVCC_FLAGS)

kernel_cu.o: kernel.cu matrix.h CNN.h CNNOnDevice.h
	$(NVCC) -c -o $@ kernel.cu $(NVCC_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
