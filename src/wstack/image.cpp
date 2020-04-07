#include <iostream>
#include <iomanip>
#include <complex>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <fftw3.h>
#include <omp.h>

#include "wstack_common.h"

// We use non-unit strides to alleviate cache thrashing effects.
    const int element_stride = 1;
    const int row_stride = 8;
    const int matrix_stride = 10;


