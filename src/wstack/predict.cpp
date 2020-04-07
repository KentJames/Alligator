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
#include "predict.h"

// We use non-unit strides to alleviate cache thrashing effects.
    const int element_stride = 1;
    const int row_stride = 8;
    const int matrix_stride = 10;

/*
  Predicts a visibility at a particular point using the direct fourier transform.
 */

std::complex<double>
predict_visibility(const std::vector<double>& points,
		   double u,
		   double v,
		   double w){

    std::complex<double> vis = {0.0,0.0};
    int npts = points.size()/2;
    for(int i = 0; i < npts; ++i){
	double l = points[2*i];
	double m = points[2*i + 1];
	double n = std::sqrt(1 - l*l - m*m) - 1.0;	
	
	std::complex<double> phase = {0,-2 * PI<double> * (u*l + v*m + w*n)};
	std::complex<double> amp = {1.0,0.0};
	vis += amp * std::exp(phase);
	
    }
    return vis;
}

std::complex<double>
predict_visibility_quantized(const std::vector<double>& points,
			     double theta,
			     double lam,
			     double u,
			     double v,
			     double w){

    double grid_size = std::floor(theta * lam);
    
    std::complex<double> vis {0.0,0.0};
    std::size_t npts = points.size()/2;

    for(std::size_t i = 0; i < npts; ++i){
	double l = points[2*i];
	double m = points[2*i + 1];
	
	//Snap the l/m to a grid point
	double lc = std::floor((l / theta + 0.5) * (double)grid_size);
	double mc = std::floor((m / theta + 0.5) * (double)grid_size);
	std::cout << std::setprecision(15);
	
	//int lc = (int)std::floor(l / theta * (double)grid_size) + grid_size/2;
	//int mc = (int)std::floor(m / theta * (double)grid_size) + grid_size/2;
	//double lq = theta * ((lc - (double)grid_size/2)/(double)grid_size);
	//double mq = theta * ((mc - (double)grid_size/2)/(double)grid_size);
	double lq = (double)lc/lam - theta/2;
	double mq = (double)mc/lam - theta/2;
	
	
	// double lq = theta * (((double)lc/(double)grid_size) - 0.5);
	// double mq = theta * (((double)mc/(double)grid_size) - 0.5);
	double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;
	
	std::complex<double> phase = {0,-2 * PI<double> * (u*lq + v*mq + w*n)};
	
	vis += 1.0 * std::exp(phase);
    }

    return vis;
}

std::vector<std::complex<double> >
predict_visibility_quantized_vec(const std::vector<double>& points,
				 double theta,
				 double lam,
				 std::vector<double> uvw){
    
    double grid_size = std::floor(theta * lam);
    
    std::vector<std::complex<double> > vis (uvw.size()/3,{0.0,0.0});
    std::size_t npts = points.size()/2;


    //#pragma omp parallel
    for (std::size_t visi = 0; visi < uvw.size()/3; ++visi){
	
	double u = uvw[3 * visi + 0];
	double v = uvw[3 * visi + 1];
	double w = uvw[3 * visi + 2];
	for(std::size_t i = 0; i < npts; ++i){
	    double l = points[2*i];
	    double m = points[2*i + 1];

	    //Snap the l/m to a grid point
	    double lc = std::floor((l / theta + 0.5) * (double)grid_size);
	    double mc = std::floor((m / theta + 0.5) * (double)grid_size);
	    std::cout << std::setprecision(15);

	    //int lc = (int)std::floor(l / theta * (double)grid_size) + grid_size/2;
	    //int mc = (int)std::floor(m / theta * (double)grid_size) + grid_size/2;
	    //double lq = theta * ((lc - (double)grid_size/2)/(double)grid_size);
	    //double mq = theta * ((mc - (double)grid_size/2)/(double)grid_size);
	    double lq = (double)lc/lam - theta/2;
	    double mq = (double)mc/lam - theta/2;

	
	    // double lq = theta * (((double)lc/(double)grid_size) - 0.5);
	    // double mq = theta * (((double)mc/(double)grid_size) - 0.5);
	    double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;

	    std::complex<double> phase = {0,-2 * PI<double> * (u*lq + v*mq + w*n)};

	    vis[visi] += 1.0 * std::exp(phase);
	}
    }
    return vis;
}




std::vector<std::complex<double>> wstack_predict(double theta,
						 double lam,
						 const std::vector<double>& points, // Sky points
						 std::vector<double> uvwvec, // U/V/W points to predict.
						 double du, // Sze-Tan Optimum Spacing in U/V
						 double dw, // Sze-Tan Optimum Spacing in W
						 int aa_support_uv,
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
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
    
    
    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double> > wstacks(oversampg,oversampg,w_planes,{0.0,0.0},element_stride,row_stride,matrix_stride);
    vector2D<std::complex<double> > skyp(oversampg,oversampg,{0.0,0.0},element_stride,row_stride);
    vector2D<std::complex<double> > plane(oversampg,oversampg,{0.0,0.0},element_stride,row_stride);
    fftw_plan plan;
    std::cout << "Planning fft's... " << std::flush;

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());    
    fftw_import_wisdom_from_filename("fftw.wisdom");

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
    
    // I'm a big boy now. Guru mode fft's~~
    plan = fftw_plan_guru_dft(2, iodims_plane, 1, iodims_howmany,
			      reinterpret_cast<fftw_complex*>(skyp.dp()),
			      reinterpret_cast<fftw_complex*>(plane.dp()),
			      FFTW_FORWARD,
			      FFTW_MEASURE);
 
    fftw_export_wisdom_to_filename("fftw.wisdom");
    std::cout << "done\n" << std::flush;
    skyp.clear();
    plane.clear();
    std::cout << "Generating sky... " << std::flush;
    generate_sky(points,skyp,theta,lam,du,dw,x0,grid_corr_lm,grid_corr_n);
    std::cout << "Sky: " << skyp(grid_size,grid_size) << "\n";
    std::cout << "done\n" << std::flush;
    fft_shift_2Darray(skyp);
    fft_shift_2Darray(wtransfer);
    multiply_fresnel_pattern(wtransfer,skyp,(std::floor(-w_planes/2)));

    std::cout << "W-Stacker: \n";
    std::cout << std::setprecision(15);

    long flops_per_vis = 6 * aa_support_uv * aa_support_uv * aa_support_w; // 3D Deconvolve
    flops_per_vis += 3 * aa_support_uv * aa_support_uv * aa_support_w; // Compute seperable kernel
    long total_flops = flops_per_vis * uvwvec.size()/3;

    std::chrono::high_resolution_clock::time_point t1_ws = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < w_planes; ++i){

	std::cout << "Processing Plane: " << i << "\n";	
	fftw_execute(plan);
	fft_shift_2Darray(plane);
	memcpy_plane_to_stack(plane,
			      wstacks,
			      grid_size,
			      i);
	multiply_fresnel_pattern(wtransfer,skyp,1);
	plane.clear();
	
    }
    std::chrono::high_resolution_clock::time_point t2_ws = std::chrono::high_resolution_clock::now();
    auto duration_ws = std::chrono::duration_cast<std::chrono::milliseconds>( t2_ws - t1_ws ).count();
    
    std::cout << "W-Stack Time: " << duration_ws << "ms \n";
   

    std::vector<std::complex<double> > visibilities(uvwvec.size()/3,{0.0,0.0});
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
#pragma omp for schedule(static,1000)
    for (std::size_t i = 0; i < uvwvec.size()/3; ++i){

    	visibilities[i] = deconvolve_visibility_(uvwvec[3*i + 0],
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
						wstacks,
						grid_conv_uv,
						grid_conv_w);
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    float duration_s = static_cast<float>(duration)/1000;
    float gflops = static_cast<float>(total_flops) / duration_s;
    std::cout << "Deconvolve Time: " << duration << "ms \n";
    std::cout << "GFLOP/s: " << gflops << "\n";
    
    //Unfortunately LLVM and GCC are woefully behind Microsoft when it comes to parallel algorithm support in the STL!!

    // std::transform(std::execution::par, uvwvec.begin(), uvwvec.end(), visibilities.begin(),
    // 		   [du,
    // 		    dw,
    // 		    aa_support_uv,
    // 		    aa_support_w,
    // 		    w_planes,
    // 		    grid_size,
    // 		    wstacks,
    // 		    grid_conv_uv,
    // 		    grid_conv_w](std::vector<double> uvw) -> std::complex<double>
    // 		   {return deconvolve_visibility(uvw,
    // 						 du,
    // 						 dw,
    // 						 aa_support_uv,
    // 						 aa_support_w,
    // 						 w_planes,
    // 						 grid_size,
    // 						 wstacks,
    // 						 grid_conv_uv,
    // 						 grid_conv_w);
    // 		   });
     
		   

    
    return visibilities;
}


//Takes lines of visibilities.
std::vector<std::vector<std::complex<double>>> wstack_predict_lines(double theta,
						       double lam,
						       const std::vector<double>& points, // Sky points
						       std::vector<std::vector<double>> uvwvec, // U/V/W points to predict.
						       double du, // Sze-Tan Optimum Spacing in U/V
						       double dw, // Sze-Tan Optimum Spacing in W
						       int aa_support_uv,
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
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
       
    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double> > wstacks(oversampg,oversampg,w_planes,{0.0,0.0}, element_stride, row_stride, matrix_stride);
    vector2D<std::complex<double> > skyp(oversampg,oversampg,{0.0,0.0}, element_stride, row_stride);
    vector2D<std::complex<double> > plane(oversampg,oversampg,{0.0,0.0}, element_stride, row_stride);
    fftw_plan plan;
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
    
    // I'm a big boy now. Guru mode fft's~~
    plan = fftw_plan_guru_dft(2, iodims_plane, 1, iodims_howmany,
			      reinterpret_cast<fftw_complex*>(skyp.dp()),
			      reinterpret_cast<fftw_complex*>(plane.dp()),
			      FFTW_FORWARD,
			      FFTW_MEASURE);
;
    fftw_export_wisdom_to_filename("fftw_l.wisdom");
    std::cout << "done\n" << std::flush;
    skyp.clear();
    plane.clear();
    std::cout << "Generating sky... " << std::flush;
    generate_sky(points,skyp,theta,lam,du,dw,x0,grid_corr_lm,grid_corr_n);    
    std::cout << "done\n" << std::flush;
    fft_shift_2Darray(skyp);
    fft_shift_2Darray(wtransfer);
    multiply_fresnel_pattern(wtransfer,skyp,(std::floor(-w_planes/2)));

    std::cout << "W-Stacker: \n";
    std::cout << std::setprecision(15);

    std::chrono::high_resolution_clock::time_point t1_ws = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < w_planes; ++i){

	std::cout << "Processing Plane: " << i << "\n";
	fftw_execute(plan);
	fft_shift_2Darray(plane);
	memcpy_plane_to_stack(plane,
			      wstacks,
			      grid_size,
			      i);
	multiply_fresnel_pattern(wtransfer,skyp,1);
	plane.clear();	
    }
    std::chrono::high_resolution_clock::time_point t2_ws = std::chrono::high_resolution_clock::now();
    auto duration_ws = std::chrono::duration_cast<std::chrono::milliseconds>( t2_ws - t1_ws ).count();
    std::cout << "W-Stack Time: " << duration_ws << "ms \n";;
    std::cout << " UVW Vec Size: " << uvwvec.size() << "\n";
    std::vector<std::vector<std::complex<double> > > visibilities(uvwvec.size());

    // To make threads play nice, pre-initialise
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    #pragma omp for schedule(dynamic)
    for (std::size_t line = 0; line < uvwvec.size(); ++ line){
	visibilities[line].resize(uvwvec[line].size()/3,0.0);
	for (std::size_t i = 0; i < uvwvec[line].size()/3; ++i){

	    visibilities[line][i] = deconvolve_visibility_(uvwvec[line][3*i + 0],
							  uvwvec[line][3*i + 1],
							  uvwvec[line][3*i + 2],
							  du,
							  dw,
							  aa_support_uv,
							  aa_support_w,
							  grid_conv_uv->oversampling,
							  grid_conv_w->oversampling,
							  w_planes,
							  grid_size,
							  wstacks,
							  grid_conv_uv,
							  grid_conv_w);
	}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    
    std::cout << "Deconvolve Time: " << duration << "ms \n";;
   
    /*
    //Unfortunately LLVM and GCC are woefully behind Microsoft when it comes to parallel algorithm support in the STL!!

    std::transform(std::execution:par, uvwvec.begin(), uvwvec.end(), visibilities.begin(),
		   [du,
		    dw,
		    aa_support_uv,
		    aa_support_w,
		    w_planes,
		    grid_size,
		    wstacks,
		    grid_conv_uv,
		    grid_conv_w](std::vector<double> uvw) -> std::complex<double>
		   {return deconvolve_visibility(uvw,
						 du,
						 dw,
						 aa_support_uv,
						 aa_support_w,
						 w_planes,
						 grid_size,
						 wstacks,
						 grid_conv_uv,
						 grid_conv_w);
		   });
    */ 
		   

    
    return visibilities;
}

