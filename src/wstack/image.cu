//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <complex>
#include <algorithm>
#include <chrono>

//CUDA Includes
#include <cuComplex.h>
#include <cufft.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"

//Thrust (CUDA STL) Includes..
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

//Our includes
#include "wstack_common.h"
#include "common_kernels.cuh"
#include "radio.cuh"
#include "image.cuh"

template <typename FloatType>
__global__ void convolve_3D(thrust::complex<FloatType> *wstacks,
			    thrust::complex<FloatType> *vis,
			    FloatType *uvec,
			    FloatType *vvec,
			    FloatType *wvec,
			    FloatType *gcf_uv,
			    FloatType *gcf_w,
			    FloatType du,
			    FloatType dw,
			    int vis_num,
			    int aa_support_uv,
			    int aa_support_w,
			    int oversampling,
			    int oversampling_w,
			    int w_planes,
			    int grid_size,
			    int oversampg){

    if(vis_num == 0) return;
    const int aa_h = aa_support_uv/2;
    const int aaw_h = aa_support_w/2;

    for(uint i = threadIdx.x; i < aa_support_uv * aa_support_uv * aa_support_w; i += blockDim.x){


	int myW = i / (aa_support_uv * aa_support_uv);
	int myV = (i % (aa_support_uv * aa_support_uv)) / aa_support_uv;
	int myU = i % aa_support_uv;

	thrust::complex<FloatType> sum = {0.0,0.0};

	for(int vi = 0; vi < vis_num; ++vi){

	    FloatType ur = uvec[vi];
	    FloatType vr = vvec[vi];
	    FloatType wr = wvec[vi];

	    int ui = cuda_ceil(ur/du) + grid_size - aa_h;
	    int vi = cuda_ceil(vr/du) + grid_size - aa_h;
	    int wi = cuda_ceil(wr/dw) + w_planes/2 - aaw_h;

	    FloatType flu = ur - cuda_ceil(ur/du)*du;
	    FloatType flv = vr - cuda_ceil(vr/du)*du;
	    FloatType flw = wr - cuda_ceil(wr/dw)*dw;

	    int ovu = static_cast<int>(cuda_floor(cuda_fabs(flu)/du * oversampling));
	    int ovv = static_cast<int>(cuda_floor(cuda_fabs(flv)/du * oversampling));
	    int ovw = static_cast<int>(cuda_floor(cuda_fabs(flw)/dw * oversampling_w));   
	    
	    int myConvU = (ui - myU) % aa_support_uv;
	    int myConvV = (vi - myV) % aa_support_uv;
	    int myConvW = (wi - myW) % aa_support_w;

	    if (myConvU < 0) myConvU += aa_support_uv;
	    if (myConvV < 0) myConvV += aa_support_uv;
	    if (myConvW < 0) myConvW += aa_support_w;

	    int myGridU = ui + myConvU;
	    int myGridV = vi + myConvV;
	    int myGridW = wi + myConvW;

	    
	    if( myGridU != myU ||
		myGridV != myV ||
		myGridW != myW ){

		if( myGridU < 0 ||
		     myGridV < 0 ||
		     myGridW < 0 ||
		     myGridU >= oversampg ||
		     myGridV >= oversampg ||
		     myGridW >= w_planes) return;

		// Complex Atomic Add here
		int mem_offset = 2 * // 2 because grid is interleaved complex array (real/imag/real/imag..)
		    (myW * oversampg * oversampg +
		     myV * oversampg + myU);

		atomicAdd(((FloatType *)wstacks + mem_offset)    ,sum.real());
		atomicAdd(((FloatType *)wstacks + mem_offset + 1),sum.imag());
		
		thrust::complex<FloatType> zero = {0.0, 0.0};
		sum = zero;
		myU = myGridU;
		myV = myGridV;
		myW = myGridW;		
	    }

	    FloatType kernval = gcf_uv[aa_support_uv * ovu + myConvU] *
		                gcf_uv[aa_support_uv * ovv + myConvV] *
		                gcf_w[aa_support_w * ovw + myConvW];

	    sum += vis[vi] * kernval;

	}
	int mem_offset = (myW * oversampg * oversampg +
			  myV * oversampg + myU);

	
	atomicAdd(((FloatType *)wstacks + mem_offset)    ,sum.real());
	atomicAdd(((FloatType *)wstacks + mem_offset + 1),sum.imag());
    }
}

template <typename FloatType>
__global__ void add_image(thrust::complex<FloatType> *in,
			  thrust::complex<FloatType> *out,
			  int n){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < n && y < n){
	thrust::complex<FloatType> res = in[y * n + x] + out[y * n + x];
	out[y * n + x] = res;
    }
}


//Elementwise multiplication of subimg with fresnel.
template <typename FloatType>
__global__ void fresnel_sky_mul(thrust::complex<FloatType> *sky,
				thrust::complex<FloatType> *fresnel,
				int n,
				int wp){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x < n && y < n){
	thrust::complex<FloatType> pown = {wp,0.0};
	thrust::complex<FloatType> fresnelv = fresnel[y*n + x];
	thrust::complex<FloatType> wtrans = thrust::pow(fresnelv, pown);
	thrust::complex<FloatType> skyv = sky[y*n + x];
	
	thrust::complex<FloatType> test = {0.0, 0.0};

	//	sky[y*n + x] = sky[y*n + x] * wtrans;
	
	if(wp == 1){
	    sky[y*n + x] = skyv * fresnelv;
	} else {
	    if (fresnelv == test){
		sky[y*n+x] = 0.0;
	    } else {
		sky[y*n+x] = skyv * wtrans;
	    }
	}
    }
}

template <typename FloatType>
__global__ void grid_correct_sky(thrust::complex<FloatType> *sky,
				 FloatType theta,
				 FloatType lam,
				 FloatType du,
				 FloatType dw,
				 int aa_support_uv,
				 int aa_support_w,
				 FloatType x0,
				 FloatType *grid_corr_lm,
				 int grid_corr_lm_size,
				 int grid_corr_lm_oversampling,
				 FloatType *grid_corr_n,
				 int grid_corr_n_size,
				 int grid_corr_n_oversampling){

    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    FloatType sky_bound_low_theta  = - theta/2;
    FloatType sky_bound_high_theta =   theta/2;

    int grid_size = static_cast<int>(cuda_floor(theta * lam));
    FloatType x0ih = round(0.5/x0);
    int oversampg = static_cast<int>(round(x0ih * grid_size));

    int lm_size_t = grid_corr_lm_size * grid_corr_lm_oversampling;
    int n_size_t = grid_corr_n_size * grid_corr_n_oversampling;
    FloatType lm_step = 1.0/(FloatType)lm_size_t;
    FloatType n_step = 1.0/(FloatType)n_size_t;
    FloatType theta_scale = x0ih * theta / oversampg;

    int true_y = y - grid_size;
    int true_x = x - grid_size;

    FloatType l = theta_scale * true_x;
    FloatType m = theta_scale * true_y;

    int lc = static_cast<int>(cuda_floor((l / theta + 0.5) * static_cast<FloatType>(grid_size)));
    int mc = static_cast<int>(cuda_floor((m / theta + 0.5) * static_cast<FloatType>(grid_size)));

    FloatType lq = (FloatType)lc/lam - theta/2;
    FloatType mq = (FloatType)mc/lam - theta/2;
    FloatType n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;
    
    int aau = std::floor((du*lq)/lm_step) + lm_size_t/2;
    int aav = std::floor((du*mq)/lm_step) + lm_size_t/2;
    int aaw = std::floor((dw*n)/n_step) + n_size_t/2;

    FloatType a = 1.0;
    a *= grid_corr_lm[aau];
    a *= grid_corr_lm[aav];
    a *= grid_corr_n[aaw];

    if( (l >= sky_bound_low_theta && l < sky_bound_high_theta) &&
	(m >= sky_bound_low_theta && m < sky_bound_high_theta)){
	thrust::complex<FloatType> source = sky[y * oversampg + x];
	source = source/a;
	sky[y * oversampg + x] = source;
    } else {
	thrust::complex<FloatType> zero = {0.0,0.0};
	sky[y * oversampg + x] = zero;
    }	    
}
				 


__host__ vector2D<std::complex<double>>
wstack_image_cu(double theta,
		double lam,
		const std::vector<std::complex<double> >& vis,
		const std::vector<double> u,
		const std::vector<double> v,
		const std::vector<double> w,
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

    // Fresnel Pattern    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    thrust::host_vector<thrust::complex<double>> wtransfer_h(oversampg*oversampg,{0.0,0.0});
    std::memcpy(wtransfer_h.data(),wtransfer.dp(),sizeof(std::complex<double>) * oversampg * oversampg);
    thrust::device_vector<thrust::complex<double>> wtransfer_d = wtransfer_h;
    thrust::device_vector<thrust::complex<double>> skyp_d(oversampg * oversampg, {0.0,0.0});
    thrust::device_vector<thrust::complex<double>> skyp_d_out(oversampg * oversampg, {0.0,0.0});
    thrust::host_vector<thrust::complex<double>> skyp_h_out(oversampg * oversampg, {0.0,0.0});
    
    
    // Work out range of W-Planes    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";

    std::cout << "Copying vis locs to GPU..." << std::flush;
    thrust::host_vector<double> uvec_h(u.size(),0.0);
    thrust::host_vector<double> vvec_h(v.size(),0.0);
    thrust::host_vector<double> wvec_h(w.size(),0.0);
    std::memcpy(uvec_h.data(),u.data(),sizeof(double) * u.size());
    std::memcpy(vvec_h.data(),v.data(),sizeof(double) * v.size());
    std::memcpy(wvec_h.data(),w.data(),sizeof(double) * w.size());    
    thrust::device_vector<double> uvec_d = uvec_h;
    thrust::device_vector<double> vvec_d = vvec_h;
    thrust::device_vector<double> wvec_d = wvec_h;
    thrust::device_vector<thrust::complex<double>> wstacks(w_planes*oversampg*oversampg,{0.0,0.0});
    std::cout << "done\n";

    std::cout << "Copying visibilities to GPU..." << std::flush;
    thrust::host_vector<thrust::complex<double>> vis_h(vis.size(),{0.0,0.0});
    std::memcpy(vis_h.data(), vis.data(), sizeof(double) * vis.size());
    thrust::device_vector<thrust::complex<double>> vis_d = vis_h;
    std::cout << "done\n";

    std::cout << "Copying convolution kernels to GPU..." << std::flush;
    std::size_t uv_conv_size = grid_conv_uv->oversampling * grid_conv_uv->size;
    std::size_t w_conv_size = grid_conv_w->oversampling * grid_conv_w->size;
    
    thrust::host_vector<double> gcf_uv_h(uv_conv_size, 0.0);
    thrust::host_vector<double> gcf_w_h(w_conv_size, 0.0);

    std::memcpy(gcf_uv_h.data(), grid_conv_uv->data, sizeof(double) * uv_conv_size);
    std::memcpy(gcf_w_h.data(), grid_conv_w->data, sizeof(double) * w_conv_size);
    thrust::device_vector<double> gcf_uv_d = gcf_uv_h;
    thrust::device_vector<double> gcf_w_d = gcf_w_h;
    std::cout << "done\n";

    // Setup FFT plan for our image/grid using cufft.
    std::cout << "Planning CUDA FFT's... " << std::flush;
    cufftHandle plan;
    cuFFTError_check(cufftPlan2d(&plan,oversampg,oversampg,CUFFT_Z2Z));
    std::cout << "done\n";

    std::size_t wstack_gs = 32;
    dim3 dimGridImage(oversampg/wstack_gs,oversampg/wstack_gs);
    dim3 dimBlockImage(wstack_gs,wstack_gs);

    // FFT Shift our Fresnel Pattern
    fft_shift_kernel <thrust::complex<double>>
	<<< dimGridImage, dimBlockImage >>>
    	((thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
    	oversampg);

    std::cout << "Starting Convolution..." << std::flush;
    dim3 dimGridConvolve(1,1,1);
    dim3 dimBlockConvolve(aa_support_uv * aa_support_uv * aa_support_w,1,1);
    std::chrono::high_resolution_clock::time_point tcs = std::chrono::high_resolution_clock::now();
    
    convolve_3D <double> <<< 1, 32 >>>
	((thrust::complex<double>*)thrust::raw_pointer_cast(wstacks.data()),
	 (thrust::complex<double>*)thrust::raw_pointer_cast(vis_d.data()),
	 (double *)thrust::raw_pointer_cast(uvec_d.data()),
	 (double *)thrust::raw_pointer_cast(vvec_d.data()),
	 (double *)thrust::raw_pointer_cast(wvec_d.data()),
	 (double *)thrust::raw_pointer_cast(gcf_uv_d.data()),
	 (double *)thrust::raw_pointer_cast(gcf_w_d.data()),
	 du, dw,
	 u.size(),
	 grid_conv_uv->size,
	 grid_conv_w->size,
	 grid_conv_uv->oversampling,
	 grid_conv_w->oversampling,
	 w_planes,
	 grid_size,
	 oversampg);
	 
	 
    cudaError_check(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point tce = std::chrono::high_resolution_clock::now();
    auto duration_convolve = std::chrono::duration_cast<std::chrono::milliseconds>(tce - tcs).count();
    std::cout << "done\n";
    std::cout << "Convolution Time: " << duration_convolve << "ms \n";

    std::cout << "Starting W-Stacking..." << std::flush;
    for(int wplane = 0; wplane < w_planes; ++wplane){
	fft_shift_kernel <thrust::complex<double>>
    	    <<< dimGridImage, dimBlockImage >>>
    	    ((thrust::complex<double>*)thrust::raw_pointer_cast(&wstacks.data()[wplane * oversampg * oversampg]),
    	     oversampg);
	
	cuFFTError_check(cufftExecZ2Z(plan,
				      (cuDoubleComplex*)thrust::raw_pointer_cast(&wstacks.data()[wplane * oversampg * oversampg]),
    				      (cuDoubleComplex*)thrust::raw_pointer_cast(skyp_d.data()),
    				      
    				      CUFFT_FORWARD));
	fresnel_sky_mul <double>
	    <<< dimGridImage, dimBlockImage >>>
	    ((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
	     (thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
	     oversampg,
	     w_planes/2 - wplane);
	add_image <double>
	    <<< dimGridImage, dimBlockImage >>>
	    ((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
	     (thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d_out.data()),
	     oversampg);
    }
    fft_shift_kernel <thrust::complex<double>>
	<<< dimGridImage, dimBlockImage >>>
	((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d_out.data()),
	 oversampg);
    cudaError_check(cudaPeekAtLastError());
    std::cout << "done\n";

    std::cout << "Applying Grid Correction..." << std::flush;
    grid_correct_sky <double>
	<<< dimGridImage, dimBlockImage >>>
	((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d_out.data()),
	 theta,
	 lam,
	 du,
	 dw,
	 grid_conv_uv->size,
	 grid_conv_w->size,
	 x0,
	 grid_corr_lm->data,
	 grid_corr_lm->size,
	 grid_corr_lm->oversampling,
	 grid_corr_n->data,
	 grid_corr_n->size,
	 grid_corr_n->oversampling);
    std::cout << "done\n";


    
    skyp_h_out = skyp_d_out;

    // Return the imaged sky.

    vector2D<std::complex<double>> sky(oversampg,oversampg,{0.0,0.0});
    std::memcpy(sky.dp(),skyp_h_out.data(),oversampg*oversampg * 2 * sizeof(double));
    return sky;

}