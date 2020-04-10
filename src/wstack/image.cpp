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
#include "image.h"


// We know the sky should be real but with the imprecision introduced
// by convolutional gridding, there will be a small imaginary component
// still.
vector2D<std::complex<double>>
wstack_image(double theta,
	     double lam,
	     const std::vector<std::complex<double> >& vis,
	     const std::vector<double>& uvwvec,
	     double du, // Sze-Tan Optimum Spacing in u/v
	     double dw, // Sze-Tan Optimum Spacing in w
	     int aa_support_uv, // Assume support is same for u and v
	     int aa_support_w,
	     double x0,
	     struct sep_kernel_data *grid_conv_uv,
	     struct sep_kernel_data *grid_conv_w,
	     struct sep_kernel_data *grid_corr_lm,
	     struct sep_kernel_data *grid_corr_n){


    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = std::round(0.5/x0);
    int oversampg = static_cast<int>(x0ih * grid_size);
    assert(oversampg > grid_size);
    //int gd = (oversampg - grid_size)/2;

    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    fft_shift_2Darray(wtransfer);
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
       
    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double>> wstacks(oversampg,oversampg,w_planes,{0.0,0.0}, element_stride, row_stride, matrix_stride);
    vector2D<std::complex<double>> sky(oversampg,oversampg,{0.0,0.0}, element_stride, row_stride);

    vector2D<std::complex<double>> image(grid_size,grid_size,{0.0,0.0}, element_stride, row_stride);
    fftw_plan *plan = (fftw_plan *)malloc(w_planes * sizeof(fftw_plan));
    std::cout << "Planning fft's... " << std::flush;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename("fftw_l.wisdom");

    fftw_iodim *iodims_plane = (fftw_iodim *)malloc(2*sizeof(fftw_iodim));
    fftw_iodim *iodims_howmany = (fftw_iodim *)malloc(sizeof(fftw_iodim));
    // Setup row dims
    iodims_plane[0].n = 2*grid_size;
    iodims_plane[0].is = 1; // Keep row elements contiguous (for now)
    iodims_plane[0].os = iodims_plane[0].is;

    // Setup matrix dims
    iodims_plane[1].n = 2*grid_size;
    iodims_plane[1].is = row_stride  + 2*grid_size*element_stride;
    iodims_plane[1].os = iodims_plane[1].is;

    // Give a unit howmany rank dimensions
    iodims_howmany[0].n = 1;
    iodims_howmany[0].is = 1;
    iodims_howmany[0].os = 1;
    
    // Here we do the transforms in-place to save memory
    for(std::size_t i = 0; i < w_planes; ++i){
	plan[i] = fftw_plan_guru_dft(2, iodims_plane, 1, iodims_howmany,
			      reinterpret_cast<fftw_complex*>(wstacks.pp(i)),
			      reinterpret_cast<fftw_complex*>(wstacks.pp(i)),
			      FFTW_BACKWARD,
			      FFTW_MEASURE);
    }
    std::cout << "Convolving 3D Visibilities...";
    std::chrono::high_resolution_clock::time_point t_convolve_start =
	std::chrono::high_resolution_clock::now();
    for(std::size_t i = 0; i < uvwvec.size()/3; ++i){
	convolve_visibility_(uvwvec[3*i + 0],
			     uvwvec[3*i + 1],
			     uvwvec[3*i + 2],
			     du,
			     dw,
			     aa_support_uv,
			     aa_support_w,
			     grid_conv_uv->oversampling,
			     grid_conv_w->oversampling,
			     w_planes,
			     grid_size,
			     vis[i],
			     wstacks,
			     grid_conv_uv,
			     grid_conv_w);
    }
    std::chrono::high_resolution_clock::time_point t_convolve_end =
	std::chrono::high_resolution_clock::now();
    std::cout << "done \n";

    auto duration_convolution =
	std::chrono::duration_cast<std::chrono::milliseconds>
	(t_convolve_end - t_convolve_start).count();

    std::cout << "Convolution Time: " << duration_convolution << "ms \n";
    

    std::cout << "W-Stacking...";
    std::chrono::high_resolution_clock::time_point t_wstack_start =
	std::chrono::high_resolution_clock::now();

    // FFT and multiply in our fresnel pattern
    for(std::size_t i = 0; i < w_planes; ++i){
	fft_shift_2Darray(wstacks,i);
	fftw_execute(plan[i]);
	multiply_fresnel_pattern(wtransfer,
				 wstacks,
				 w_planes - i,
				 i);
	// Iteratively add this back to the sky reconstruction.
	for(std::size_t v = 0; v < oversampg; ++v){
	    for(std::size_t u = 0; u < oversampg; ++u){
		sky(u,v) += wstacks(u,v,i);
	    }
	}
    }
    // Bring sky back to w=0 co-ordinate
    multiply_fresnel_pattern(wtransfer,sky,std::floor(-w_planes/2));
    fft_shift_2Darray(sky);

    // Extract the central portion of the map.
    // As Steve Gull always says - "Throw away half the map, its useless!"
    std::size_t i0 = sky.d1s()/4;
    std::size_t i1 = 3 * i0;
    std::size_t j0 = sky.d2s()/4;
    std::size_t j1 = 3 * j0;

    for(std::size_t j = j0; j < j1; ++j){
	for(std::size_t i = i0; i < i1; ++i){
	    image(i-i0,j-j0) = sky(i,j); 
	}
    }


    
    
    std::chrono::high_resolution_clock::time_point t_wstack_end =
	std::chrono::high_resolution_clock::now();
    std::cout << "done \n";


    auto duration_wstack =
	std::chrono::duration_cast<std::chrono::milliseconds>
	(t_wstack_end - t_wstack_start).count();
    std::cout << "W-Stack Time: " << duration_wstack << "ms \n";
    
    return image;

}
