CC = nvcc

CXXFLAGS = -gencode arch=compute_35,code=sm_35



all: libhdf5_cu.a

clean:
	rm *.o
	rm *.a

libhdf5_cu.a: hdf5_cu.o
	ar rcs $@ $^

hdf5_cu.o: hdf5.c hdf5_h.h
	$(CC) -c -o $@ $< -lhdf5 $(CXXFLAGS)