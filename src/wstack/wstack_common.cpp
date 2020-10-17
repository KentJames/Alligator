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

std::vector<double> generate_random_points(int npts,
					   double theta){

    std::vector<double> points(2 * npts,0.0);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta/2,theta/2);

    for(int i = 0; i < npts; ++i){
	points[2*i] = distribution(generator); // l
	points[2*i + 1] = distribution(generator); // m
    }

    return points;
}

std::vector<double> generate_testcard_dataset(double theta){

    std::vector<double> points = {0.95,0.95,-0.95,-0.95,0.95,-0.95,-0.95,0.95,0.0,0.5,0.0,-0.5,0.5,0.0,-0.5,0.0,0.0,0.0};
    std::transform(points.begin(), points.end(), points.begin(),
		   [theta](double c) -> double { return c * (theta/2);});

    for(std::size_t i = 0; i < points.size(); ++i){
	std::cout << points[i] << " ";
       
    }
    std::cout << "\n";
    return points;
}

std::vector<double> generate_testcard_dataset_simple(double theta){

    std::vector<double> points = {0.0,0.0};
    std::transform(points.begin(), points.end(), points.begin(),
		   [theta](double c) -> double { return c * (theta/2);});

    for(std::size_t i = 0; i < points.size(); ++i){
	std::cout << points[i] << " ";
       
    }
    std::cout << "\n";
    return points;
}



vector2D<std::complex<double> > generate_fresnel(double theta,
						double lam,
						double dw,
						double x0){

    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = std::round(0.5/x0);
    int oversampg = static_cast<int>(x0ih * grid_size);
    assert(oversampg > grid_size);
    int gd = (oversampg - grid_size)/2;

    std::cout << "xoih: " << x0ih << "\n";
    std::cout << "oversampg: " << oversampg << "\n";
    std::cout << "gd: " << gd << "\n";

    vector2D<std::complex<double> > wtransfer(oversampg,oversampg,{0.0,0.0});
    
    for (int y=0; y < grid_size; ++y){
	for (int x=0; x < grid_size; ++x){
	    double l = theta * ((double)x - grid_size / 2) / grid_size;
	    double m = theta * ((double)y - grid_size / 2) / grid_size;
	    double ph = dw * (1.0 - std::sqrt(1.0 - l*l - m*m));
	    
	    std::complex<double> wtrans = {0.0, 2 * PI<double> * ph};
	    int xo = x+gd;
	    int yo = y+gd;
	    wtransfer(xo,yo) = std::exp(wtrans);
	}

	
    }

    return wtransfer;
}


//Quantises our sources onto the sky.
void generate_sky(const std::vector<double>& points,
		  vector2D<std::complex<double> >& sky, // We do this by reference because of FFTW planner.
		  double theta,
		  double lam,
		  double du,
		  double dw,
		  double x0,
		  struct sep_kernel_data *grid_corr_lm,
		  struct sep_kernel_data *grid_corr_n){
    
    
    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = 0.5/x0;
    int oversampg = static_cast<int>(std::round(x0ih * grid_size));
    assert(oversampg > grid_size);
    int gd = (oversampg - grid_size)/2;

    int npts = points.size()/2;

    for (int i = 0; i < npts; ++i){

	// Calculate co-ordinates
	double l = points[2*i];
	double m = points[2*i + 1];

	int lc = static_cast<int>(std::floor((l / theta + 0.5) *
					     static_cast<double>(grid_size)));
        int mc = static_cast<int>(std::floor((m / theta + 0.5) *
					     static_cast<double>(grid_size)));
	// double lq = theta * ((static_cast<double>(lc) - (double)grid_size/2)/(double)grid_size);
	// double mq = theta * ((static_cast<double>(mc) - (double)grid_size/2)/(double)grid_size);

	double lq = (double)lc/lam - theta/2;
	double mq = (double)mc/lam - theta/2;
	double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;
	// // Calculate grid correction function

	int lm_size_t = grid_corr_lm->size * grid_corr_lm->oversampling;
	int n_size_t = grid_corr_n->size * grid_corr_n->oversampling;
	double lm_step = 1.0/(double)lm_size_t;
	double n_step = 1.0/(double)n_size_t; //Not too sure about this

	int aau = std::floor((du*lq)/lm_step) + lm_size_t/2;
	int aav = std::floor((du*mq)/lm_step) + lm_size_t/2;
	int aaw = std::floor((dw*n)/n_step) + n_size_t/2;
	
	double a = 1.0;
	a *= grid_corr_lm->data[aau];
	a *= grid_corr_lm->data[aav];
	a *= grid_corr_n->data[aaw];

	std::complex<double> source = {1.0,0.0};
	source = source / a;
	int lco = gd+lc;
	int mco = gd+mc;
	sky(lco,mco) += source; // Sky needs to be transposed, not quite sure why.
    }
    
}

// I defined this for mixing vector3D and vector2D. Ugly but does the job.
// Really would rather clean up the generic side of this over time.
void multiply_fresnel_pattern(vector2D<std::complex<double>>& fresnel,
			      vector3D<std::complex<double>>& sky,
			      int t,
			      std::size_t planei){
    assert(fresnel.size() == sky.d1s() * sky.d2s());
    std::complex<double> ft = {0.0,0.0};
    std::complex<double> st = {0.0,0.0};
    std::complex<double> test = {0.0,0.0};

    size_t grid_sizex = fresnel.d1s();
    size_t grid_sizey = fresnel.d2s();
    
    for (std::size_t j = 0; j < grid_sizey; ++j){
	for (std::size_t i = 0; i < grid_sizex; ++i){
	    ft = fresnel(i,j);
	    st = sky(i,j,planei);	
	
	    if (t == 1){
		sky(i,j,planei) = st * ft;
	    } else {
	    
		if (ft == test) continue; // Otherwise std::pow goes a bit fruity
		sky(i,j,planei) = st  * std::pow(ft,t);
	    }
	}
    }
}


void multiply_fresnel_pattern(vector2D<std::complex<double>>& fresnel,
			      vector2D<std::complex<double>>& sky,
			      int t){
    assert(fresnel.size() == sky.size());
    std::complex<double> ft = {0.0,0.0};
    std::complex<double> st = {0.0,0.0};
    std::complex<double> test = {0.0,0.0};

    std::size_t grid_sizex = fresnel.d1s();
    std::size_t grid_sizey = fresnel.d2s();
    
    for (std::size_t j = 0; j < grid_sizey; ++j){
	for (std::size_t i = 0; i < grid_sizex; ++i){
	    ft = fresnel(i,j);
	    st = sky(i,j);	
	
	    if (t == 1){
		sky(i,j) = st * ft;
	    } else {
	    
		if (ft == test) continue; // Otherwise std::pow goes a bit fruity
		sky(i,j) = st  * std::pow(ft,t);
	    }
	}
    }
}

// void zero_pad_2Darray(const vector3D<std::complex<double>>& array,
// 		      vector2D<std::complex<double>>& padded_array,
// 		      double x0,
// 		      std::size_t planei){

//     int x0i = static_cast<int>(std::round(1.0/x0));
//     std::cout << "xoi: " << x0i << "\n";
//     int i0 = padded_array.d1s()/x0i;
//     int i1 = 3*(padded_array.d1s()/x0i);
//     int j0 = padded_array.d2s()/x0i;
//     int j1 = 3*(padded_array.d2s()/x0i);
//     for(int j = j0; j < j1; ++j){
// 	for(int i = i0; i < i1; ++i){
// 	    padded_array(i,j) = array(i-i0,j-j0,planei);
// 	}
//     }
// }


void zero_pad_2Darray(const vector2D<std::complex<double>>& array,
		      vector2D<std::complex<double>>& padded_array,
		      double x0){

    int x0i = static_cast<int>(std::round(1.0/x0));
    std::cout << "xoi: " << x0i << "\n";
    int i0 = padded_array.d1s()/x0i;
    int i1 = 3*(padded_array.d1s()/x0i);
    int j0 = padded_array.d2s()/x0i;
    int j1 = 3*(padded_array.d2s()/x0i);
    for(int j = j0; j < j1; ++j){
	for(int i = i0; i < i1; ++i){
	    padded_array(i,j) = array(i-i0,j-j0);
	}
    }
}


void memcpy_plane_to_stack(vector2D<std::complex<double>>&plane,
			   vector3D<std::complex<double>>&stacks,
			   std::size_t grid_size,
			   std::size_t planei,
			   std::size_t direction){

    //Calculate memory copy amount based on striding information.
    //Assume strides for n=1 and n=2 dimensions are the same between
    //stacks and plane. If they aren't then can't use memcpy directly.

    std::size_t p1s,p2s,s1s,s2s,p1d,p2d,s1d,s2d;//,s3s;


    p1d = plane.d1s();
    p2d = plane.d2s();
    s1d = stacks.d1s();
    s2d = stacks.d2s();

    // Make sure dimensions are the same
    assert(p1d == s1d);
    assert(p2d == s2d);
   
    
    p1s = plane.s1s();
    p2s = plane.s2s();
    s1s = stacks.s1s();
    s2s = stacks.s2s();
    //s3s = stacks.s3s();
    
    // Let us really make sure the strides are the same
    assert(p1s == s1s);
    assert(p2s = s2s);

    

    std::size_t copy_size = (p1d*p1s + p2s) * p2d * sizeof(std::complex<double>); 

    std::complex<double> *wp = stacks.pp(planei);  
    std::complex<double> *pp = plane.dp();

    if (direction){
	std::memcpy(wp,pp,copy_size);
    } else {
	std::memcpy(pp,wp,copy_size);
    }

}
