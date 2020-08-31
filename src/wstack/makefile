include ../../user.mk

ifndef CUDA
override CXXFLAGS += -fopenmp -std=c++14 -I../common
else
override CXXFLAGS += -g -fopenmp -DCUDA_ACCELERATION -std=c++14 -I../common -I/usr/local/cuda/include 
endif

ifndef CUDA
override LDFLAGS +=  -lm  -L../common -lgomp -lhdf5_r -lfftw3_threads -lfftw3 -lhdf5 
else
override LDFLAGS += -L../common -lgomp -lhdf5_cu -lfftw3_threads -lfftw3 -lm -lhdf5 -L$(CUDA_LIBDIR) -lcuda -lcudart -lcufft 
endif

override CUDA_FLAGS += -std=c++14  -use_fast_math -maxrregcount=64 -I../common -gencode arch=compute_60,code=sm_60

BINDIR = ./

all: wstack.out


clean:
	rm -f *.o 
	rm -f *.out

ifndef CUDA
wstack.out: wstack.o wstack_common.o libwstack.a
	$(CXX) wstack.o wstack_common.o -o $@  -L ./ -lwstack $(LDFLAGS)
	cp $@ ../../
else
wstack.out: wstack.o wstack_common.o libwstack.a
	$(NVCC) wstack.o wstack_common.o -o $@  -L ./ -lwstack $(LDFLAGS)
	cp $@ ../../
endif

wstack.o: wstack.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

wstack_common.o: wstack_common.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

predict.o: predict.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

image.o: image.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)


ifndef CUDA
libwstack.a: predict.o image.o wstack_common.o
	ar rcs $@ $^ 
else
predict_cu.o: predict.cu
	$(NVCC) -c $^ -o $@ $(CUDA_FLAGS)

image_cu.o: image.cu
	$(NVCC) -c $^ -o $@ $(CUDA_FLAGS)

libwstack.a: predict.o predict_cu.o image.o image_cu.o wstack_common.o
	ar rcs $@ $^
endif
