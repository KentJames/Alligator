CC	      = gcc
CXX           = g++
NVCC          = nvcc
LINKER        = g++
CPPFLAGS      ?=
CXXFLAGS      ?= -O3 -Wall -pedantic -I/opt/homebrew/include/
NVCCFLAGS     ?= -O3 -Xcompiler "-Wall" #-Xptxas -v
LDFLAGS       ?= -L/opt/homebrew/lib/
DOXYGEN       ?= doxygen
PYBUILDFLAGS   ?=
PYINSTALLFLAGS ?=

#GPU_ARCHS     ?= 30 32 35 37 50 52 53 # Nap time!
#GPU_ARCHS     ?= 35 52
GPU_ARCHS     ?= 35 61

CUDA_HOME     ?= /usr/local/cuda
CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

#CUDA = 1 # Enables CUDA Acceleration
