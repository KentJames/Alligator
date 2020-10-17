#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <complex>
#include <random>
#include <numeric>

//CUDA Includes

#include "helper_string.h"
#include "wstack_common.h"
#include "predict.h"
#include "image.h"
#ifdef CUDA_ACCELERATION
#include "predict.cuh"
#include "image.cuh"
#endif

void showHelp(){

    std::cout<<"\ncuStacker v0.1\n";
    std::cout<<"James Kent <jck42@cam.ac.uk>\n";
    std::cout<<"A W-Stacking Implementation originally devised by Sze-Tan \n";
    std::cout<<"University of Cambridge, 2020\n\n";

    std::cout<<"General Parameters: \n";
    std::cout<<"\t-theta               Field of View (Radians)\n";
    std::cout<<"\t-lambda              Number of Wavelengths\n";
    std::cout<<"\t-npts                Number of random point sources(Default:10)\n";
    std::cout<<"\t-mode                Predict(0) or Image (1)\n\n";

    std::cout<<"Sze Tan Parameters: \n";
    std::cout<<"\t-sepkern_uv          Seperable Sze-Tan Grid Convolution Kernels for U/V\n";
    std::cout<<"\t-sepkern_w           Seperable Sze-Tan Grid Convolution Kernels for W\n";
    std::cout<<"\t-sepkern_lm          Seperable Sze-Tan Grid Correction Kernels for l/m\n";
    std::cout<<"\t-sepkern_n           Seperable Sze-Tan Grid Correction Kernels for n\n\n";

    std::cout<<"Predict Parameters: \n";
    std::cout<<"\t-predict_file        CSV File giving U/V/W Files to predict\n";
    std::cout<<"\t-pu                  Predict u value\n";
    std::cout<<"\t-pv                  Predict v value\n";
    std::cout<<"\t-pw                  Predict w value\n\n";

    std::cout<<"Compute Parameters: \n";
    std::cout<<"\t-cuda                CPU(0) or CUDA(1)\n";
    std::cout<<"\t-device              GPU Device to use (Default: 0)\n";

}

int main(int argc, char **argv) {

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {

	showHelp();
	return 0;
    }
    
#ifdef CUDA_ACCELERATION
    //Get information on GPU's in system.
    std::cout << "CUDA System Information: \n\n";
    int cuda_acceleration = 0;
    int numberofgpus;
    int dev_no = 0;
    

  
    cudaGetDeviceCount(&numberofgpus);
    std::cout << " Number of GPUs Detected: " << numberofgpus << "\n\n";

    cudaDeviceProp *prop = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * numberofgpus);
  
    for(int i=0; i<numberofgpus;i++){


	cudaGetDeviceProperties(&prop[i],i);
    
	std::cout << "\tDevice Number: " << i <<" \n";
	std::cout << "\t\tDevice Name: " << prop->name <<"\n";
	std::cout << "\t\tTotal Memory: " << (double)prop->totalGlobalMem / (1024 * 1024) << " MB \n";
	std::cout << "\t\tShared Memory (per block): " << (double)prop->sharedMemPerBlock / 1024 << " kB \n";
	std::cout << "\t\tClock Rate: " << prop->clockRate << "\n";
	std::cout << "\t\tMultiprocessors: " << prop->multiProcessorCount << "\n";
	std::cout << "\t\tThreads Per MP: " << prop->maxThreadsPerMultiProcessor << "\n";
	std::cout << "\t\tThreads Per Block: " << prop->maxThreadsPerBlock << "\n";
	std::cout << "\t\tThreads Per Dim: " << prop->maxThreadsDim << "\n";
	std::cout << "\t\tThreads Per Warp: " << prop->warpSize << "\n";
	std::cout << "\t\tUnified Addressing: " << prop->unifiedAddressing << "\n";
	std::cout << "\n";
         
    }


    if (checkCmdLineFlag(argc, (const char **) argv, "cuda")){
	cuda_acceleration = getCmdLineArgumentInt(argc, (const char **) argv, "cuda");
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "device")){
	dev_no = getCmdLineArgumentInt(argc, (const char **) argv, "device");
    }

    cudaSetDevice(dev_no);
#endif

    
    double theta = false;
    double lambda = false;
    int mode = 0;
    int npts = 0;
    int random_points = 0;
    double pu, pv, pw;
    pu = pv = pw = 0;

  

    char *sepkern_uv_file = NULL;
    char *sepkern_w_file = NULL;
    char *sepkern_lm_file = NULL;
    char *sepkern_n_file = NULL;

    
  
    if (checkCmdLineFlag(argc, (const char **) argv, "theta") == 0 ||
	checkCmdLineFlag(argc, (const char **) argv, "lambda") == 0) {
	std::cout << "No theta or lambda specified!\n";
	showHelp();
	return 0;
    }
    else {
	theta =  getCmdLineArgumentDouble(argc, (const char **) argv, "theta");
	lambda = getCmdLineArgumentDouble(argc, (const char **) argv, "lambda");
	if (theta < 0) {
	    std::cout << "Invalid theta value specified\n";
	    return 0;
	}
	if (lambda < 0) {
	    std::cout << "Invalid lambda value specified\n";
	    return 0;
	}
    }    

    if(checkCmdLineFlag(argc, (const char**) argv, "npts")){
	npts = getCmdLineArgumentInt(argc, (const char **) argv, "npts");
    }

    if(checkCmdLineFlag(argc, (const char**) argv, "random_points")){
        random_points = getCmdLineArgumentInt(argc, (const char **) argv, "random_points");
    }

    

    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_uv") == 0){

	std::cout << "No U/V Sze-Tan Grid Convolution Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_uv", &sepkern_uv_file);

    } 


    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_w") == 0){

	std::cout << "No W Sze-Tan Grid Convolution Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_w", &sepkern_w_file);

    } 

    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_lm") == 0){

	std::cout << "No l/m Sze-Tan Grid Correction Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_lm", &sepkern_lm_file);

    } 


    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_n") == 0){

	std::cout << "No n Sze-Tan Grid Correction Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_n", &sepkern_n_file);

    } 
  

    if (checkCmdLineFlag(argc, (const char **) argv, "mode") == 0){
	std::cout << "No mode specified!\n";
	showHelp();
	return 0;
    }
    else {
	mode = getCmdLineArgumentDouble(argc, (const char **) argv, "mode");
	if (mode < 0) {
	    std::cout << "Invalid theta value specified\n";
	    return 0;
	}      
    }


    struct sep_kernel_data sepkern_uv;
    struct sep_kernel_data sepkern_w;
    struct sep_kernel_data sepkern_lm;
    struct sep_kernel_data sepkern_n;

#ifdef CUDA_ACCELERATION
    if(cuda_acceleration){

	std::cout << "Loading Kernel...";
	load_sep_kern_CUDA_T(sepkern_uv_file,&sepkern_uv);
	std::cout << "Loading W Kernel...";
	load_sep_kern_CUDA_T(sepkern_w_file,&sepkern_w);
	
	std::cout << "Loading AA Kernel...";
	load_sep_kern_CUDA(sepkern_lm_file, &sepkern_lm);
	std::cout << "Loading AA Kernel...";
	load_sep_kern_CUDA(sepkern_n_file, &sepkern_n);
	
    } else {
#endif
	std::cout << "Loading Kernel...";
	load_sep_kern_T(sepkern_uv_file,&sepkern_uv);
	std::cout << "Loading W Kernel...";
	load_sep_kern_T(sepkern_w_file,&sepkern_w);
	
	std::cout << "Loading AA Kernel...";
	load_sep_kern(sepkern_lm_file, &sepkern_lm);
	std::cout << "Loading AA Kernel...";
	load_sep_kern(sepkern_n_file, &sepkern_n);
#ifdef CUDA_ACCELERATION
    } 
#endif
    double du = sepkern_uv.du;
    double dw = sepkern_w.dw;
    double x0 = sepkern_lm.x0;

    int support_uv = sepkern_uv.size;
    int support_w = sepkern_w.size;

    std::cout << "Optimum Parameters: " << "\n";
    std::cout << "dw: " << dw << "\n";
    std::cout << "du: " << du << "\n";
   
    if (checkCmdLineFlag(argc, (const char **) argv, "pu") == 0 ||
	checkCmdLineFlag(argc, (const char **) argv, "pv") == 0 ||
	checkCmdLineFlag(argc, (const char **) argv, "pw") == 0) {
	std::cout << "No u/v/w co-ordinates specified for predict!\n";
	showHelp();	  
	return 0;
    }
    else {
	pu =  getCmdLineArgumentDouble(argc, (const char **) argv, "pu");
	pv =  getCmdLineArgumentDouble(argc, (const char **) argv, "pv");
	pw =  getCmdLineArgumentDouble(argc, (const char **) argv, "pw");
	}
    
    std::vector<std::complex<double>> visq;
    std::vector<std::complex<double>>  vis;
    std::vector<double> points = generate_testcard_dataset(theta);
    //std::vector<double> points = {0.1,0.1};
    std::vector<double> uvec;
    std::vector<double> vvec;
    std::vector<double> wvec;
    std::vector<double> uvwvec(3*npts,0.0);;
    std::cout << "Rand points: " << random_points << "\n";
    if(random_points == 0){
	
	uvec.resize(npts);
	std::fill(uvec.begin(), uvec.end(),pu);
	
	vvec.resize(npts);
	std::fill(vvec.begin(), vvec.end(),pv);
	
	wvec.resize(npts);
	std::fill(wvec.begin(), wvec.end(),pw);
	
    } else if(mode == 1) {
	    
	uvec = generate_random_visibilities_1D_uv_gaussian(theta,lambda,npts);
	vvec = generate_random_visibilities_1D_uv_gaussian(theta,lambda,npts);
	wvec = generate_random_visibilities_1D_w_gaussian(theta,lambda,dw,npts);
	    
    } else {
	uvec = generate_random_visibilities_1D_uv(theta,lambda,npts);
	vvec = generate_random_visibilities_1D_uv(theta,lambda,npts);
	wvec = generate_random_visibilities_1D_w(theta,lambda,dw,npts);
    }
	    
	
    for (std::size_t i = 0; i < uvec.size(); ++i){
	uvwvec[3*i + 0] = uvec[i];
	uvwvec[3*i + 1] = vvec[i];
	uvwvec[3*i + 2] = wvec[i];	    
    }

	
    std::cout << "Generating DFT visibilities from sky point source model..." << std::flush;
    visq = predict_visibility_quantized_vec(points,theta,lambda,uvwvec);
    std::cout << "done\n" << std::flush;


    double x0ih = std::round(0.5/x0);
    int grid_size = static_cast<int>(std::floor(theta*lambda));
    int oversampg = static_cast<int>(x0ih * grid_size);
    
    /* 
       PREDICT
    */
    if (mode == 0){

#ifdef CUDA_ACCELERATION
	if (cuda_acceleration){

	    // Only supports 8x8x4 convolution kernel at present
	    vis = wstack_predict_cu_3D(theta,
				       lambda,
				       points,
				       uvec,
				       vvec,
				       wvec,
				       du, dw,
				       support_uv,
				       support_w,
				       x0,
				       &sepkern_uv,
				       &sepkern_w,
				       &sepkern_lm,
				       &sepkern_n);
				    
	} else {
#endif	    
	    
	    vis = wstack_predict(theta,
				 lambda,
				 points,
				 uvwvec,
				 du,
				 dw,
				 support_uv,
				 support_w,
				 x0,
				 &sepkern_uv,
				 &sepkern_w,
				 &sepkern_lm,
				 &sepkern_n);
#ifdef CUDA_ACCELERATION
	}
#endif	


	std::vector<double> error (visq.size(), 0.0);
	std::transform(vis.begin(),vis.end(),visq.begin(),error.begin(),
		       [npts](std::complex<double> wstack,
			      std::complex<double> dft)
		       -> double { return std::abs(dft-wstack);});
    
	// for(int i = 0; i < 32; ++ i){
    
	std::cout << "Example Vis: " << vis[0] << "\n" << std::flush;
	std::cout << "Example DFT Vis: " << visq[0] << "\n" << std::flush;
	std::cout << "Example Error: " << error[0] << "\n" << std::flush;
	// }
	//std::for_each(error.begin(), error.end(), [](const double n) { std::cout << n << " ";});
	double agg_error = std::accumulate(error.begin(), error.end(),0.0);
	std::cout << "Aggregate Error: " << agg_error<<"\n";
	std::cout << "Aggregate Error / Npts: " << agg_error/error.size()<<"\n";
    }
    /* 
       IMAGE
    */
    else {
	vector2D<std::complex<double>> wstack_sky;
#ifdef CUDA_ACCELERATION
	if(cuda_acceleration){
	    wstack_sky = wstack_image_cu(theta,
					 lambda,
					 visq,
					 uvec,
					 vvec,
					 wvec,
					 du,
					 dw,
					 support_uv,
					 support_w,
					 x0,
					 &sepkern_uv,
					 &sepkern_w,
					 &sepkern_lm,
					 &sepkern_n);
			    

	} else {
#endif
	    //for(int i = 0; i < uvwvec.size(); ++i) uvwvec[i] = 0.0;
	    //std::complex<double> visz = {1.0,0.0};
	    //for(int i = 0; i < visq.size(); ++i) visq[i] = visz;
	    std::cout << "Visq example: " << visq[0] << "\n";
	    wstack_sky = wstack_image(theta,
				      lambda,
				      visq,
				      uvwvec,
				      du,
				      dw,
				      support_uv,
				      support_w,
				      x0,
				      &sepkern_uv,
				      &sepkern_w,
				      &sepkern_lm,
				      &sepkern_n);
	    
#ifdef CUDA_ACCELERATION
	}
#endif
		    std::cout << "Wstack size: " <<  wstack_sky.size() << "\n";
		    std::cout << "Wstack Value at (0,0): " << wstack_sky(oversampg/2,oversampg/2).real() <<  " " << wstack_sky(oversampg/2,oversampg/2).imag() << "\n";
	    
	    // Normalise
	    for(std::size_t i = 0; i < wstack_sky.size(); ++i) wstack_sky(i) /= (float)visq.size();
	    std::cout << "Normalised Wstack Value at (0,0): " << wstack_sky(oversampg/2,oversampg/2) << "\n";
	    std::cout << "Normalised Wstack Value at (0,1): " << wstack_sky(oversampg/2,oversampg/2+1) << "\n";
	    std::cout << "Normalised Wstack Value at (0,2): " << wstack_sky(oversampg/2,oversampg/2+2) << "\n";
	    std::cout << "Normalised Wstack Value at (0,3): " << wstack_sky(oversampg/2,oversampg/2+3) << "\n";
	    std::cout << "Normalised Wstack Value at (256,256): " << wstack_sky(oversampg/2+256,oversampg/2+256) << "\n";
	    std::cout << "Normalised Wstack Value at (0,512): " << wstack_sky(oversampg/2,oversampg/2+512) << "\n";
	    std::cout << "Normalised Wstack Value at (128,0): " << wstack_sky(oversampg/2 + 128,oversampg/2) << "\n";

	{
		std::ofstream file("wstack_sky.out", std::ios::binary);
		double *row = (double*)malloc(sizeof(double) * oversampg);
		for(int i = 0; i < oversampg; ++i){
		    for(int j = 0; j < oversampg; ++j){
			row[j] = wstack_sky(i,j).real();
		    }
		    file.write(reinterpret_cast<char*>(row), sizeof(double) * oversampg );
		}

	}
	    	
    }

    
    
    return 0;
}
