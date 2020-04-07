#include "hdf5.cuh"

#ifndef WSTACK_H
#define WSTACK_H

#define THREADS_BLOCK 16

#include <vector>
#include <complex>
#include <chrono>
#include <algorithm>
#include <random>

template <class T>
constexpr T PI = T(3.1415926535897932385L);
/*
  The purpose of these templates is to implement a lightweight numerical type agnostic way of
  doing multidimensional array access and operations. A good implementation seems
  lacking in C++. 

  I'd rather have avoided doing it but here we are. It's basically just syntactic
  sugar defined around a normal std::vector

  It allows arbitrary striding and data shapes.
*/
template <typename T>
class vector2D {
public:
    /* 
       d1 = n=1 dimension size
       d2 = n=2 dimension size
       t = datatype
       s1 = Distance (in terms of units of T) in between contiguous array elements
       s2 = Offset (in terms of units of T) between row elements.

       I decided against byte striding because it can decrease performance.
       Basically I'd prefer raw compute performance over memory performance.
     */
    vector2D(size_t d1=0, size_t d2=0, T const & t=T(), size_t s1 = 1, size_t s2 = 0) :
        d1(d1), d2(d2), s1(s1), s2(s2), data(d2*d1*s1 + d2*s2, t)
    {}

    ~vector2D(){
	
    }
    
    T & operator()(size_t i, size_t j) {
	//return data[j*d1 + i];
	return data[j*(d1*s1+s2) + i*s1];
    }

    T const & operator()(size_t i, size_t j) const {
	//return data[j*d1 + i];
	return data[j*(d1*s1+s2) + i*s1];
    }
    
    T & operator()(size_t i) {
        return data[i];
    }

    T const & operator()(size_t i) const {
        return data[i];
    }

    size_t size(){ return (d1*d2); }
    size_t d1s(){ return d1; }
    size_t d2s(){ return d2; }
    size_t s1s(){ return s1; }
    size_t s2s(){ return s2; }
    T* dp(){ return data.data();}

    void clear(){
	std::fill(data.begin(), data.end(), 0);
    }

    void transpose(){

	std::vector<T> datat((d1*s1)*(d2+s2),0);
	for(int j = 0; j < d1; ++j){
	    for(int i = 0; i < d2; ++i){
		data[i*(d2*s1+s2) + j*s1] = data[j*(d1*s1+s2) + i*s1];
	    }
	}
	free(data);
	data = datat;
    }

private:
    size_t d1,d2; // Data Size
    size_t s1,s2; // Stride
    std::vector<T> data;
};


template <typename T>
class vector3D {
public:
    /* 

       d1 = n=1 dimension size
       d2 = n=2 dimension size
       d3 = n=3 dimension size
       t = datatype
       s1 = Distance (in terms of units of T) in between contiguous array elements
       s2 = Offset (in terms of units of T) between row elements.
       s3 = Offset (in terms of units of T) between matrix elements.

       I decided against byte striding because it can decrease performance.
       Basically I'd prefer raw compute performance over memory performance.

     */
    vector3D(size_t d1=0, size_t d2=0, size_t d3=0,
	//      T const & t=T(), size_t s1=1, size_t s2=0, size_t s3=0) :
	// d1(d1+s2), d2(d2+s3), d3(d3),
	// s1(s1), s2(s2), s3(s3), data(d1*d2*d3 + d1*s2 + d2*s3, t)
	     T const & t=T(), size_t s1=1, size_t s2=0, size_t s3=0) :
        d1(d1), d2(d2), d3(d3),
	s1(s1), s2(s2), s3(s3),
	data(d3*(d2*(d1*s1 +s2)+s3), t)
    {
	n3s = d2*(d1*s1+s2)+s3;
	n2s = d1*s1+s2;
    }

    ~vector3D(){
	
    }

    T & operator()(size_t i, size_t j, size_t k) {
	//return data[k*d1*d2 + j*d1 + i];
        return data[k*n3s + j*n2s + i*s1];
    }

    T const & operator()(size_t i, size_t j, size_t k) const {
	//return data[k*d1*d2 + j*d1 + i];
        return data[k*n3s + j*n2s + i*s1];
    }

    T & operator()(size_t i) {
        return data[i];
    }

    T const & operator()(size_t i) const {
        return data[i];
    }

    size_t size(){ return d1*d2*d3; }
    size_t d1s(){ return d1; }
    size_t d2s(){ return d2; }
    size_t d3s(){ return d3; }
    size_t s1s(){ return s1; }
    size_t s2s(){ return s2; }
    size_t s3s(){ return s3; }

    
    T* dp(){ return data.data();}
    T* pp(std::size_t planei){
	return data.data() + planei*n3s;
    }

    void clear(){
	std::fill(data.begin(), data.end(), 0);
    }

    void fill_random(){
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator;
	generator.seed(seed);
	std::uniform_real_distribution<double> distribution(0,1);
	auto gen = [&distribution, &generator](){
                   return distribution(generator);
               };
	std::generate(std::begin(data), std::end(data), gen);

    }

private:
    size_t d1,d2,d3;
    size_t s1,s2,s3;
    size_t n2s,n3s;
    
    std::vector<T> data;
};




void multiply_fresnel_pattern(vector2D<std::complex<double>>& fresnel,
			      vector2D<std::complex<double>>& sky,
			      int t);
void zero_pad_2Darray(const vector2D<std::complex<double>>& array,
		      vector2D<std::complex<double>>& padded_array,
		      double x0);

void fft_shift_2Darray(vector2D<std::complex<double>>& array);

void memcpy_plane_to_stack(vector2D<std::complex<double>>&plane,
			   vector3D<std::complex<double>>&stacks,
			   std::size_t grid_size,
			   std::size_t planei);

vector2D<std::complex<double>>
generate_fresnel(double theta,
		 double lam,
		 double dw,
		 double x0);

//Quantises our sources onto the sky.
void generate_sky(const std::vector<double>& points,
		  vector2D<std::complex<double>>& sky, // We do this by reference because of FFTW planner.
		  double theta,
		  double lam,
		  double du,
		  double dw,
		  double x0,
		  struct sep_kernel_data *grid_corr_lm,
		  struct sep_kernel_data *grid_corr_n);

std::vector<double>generate_random_points(int npts, double theta);
std::vector<double> generate_testcard_dataset(double theta);
std::vector<double> generate_testcard_dataset_simple(double theta);

// Various inlines for generating visibilities, mostly useful for testing

static inline std::vector<std::vector<double>> generate_random_visibilities(double theta,
							      double lambda,
							      double dw,
							      int npts){

    
    std::vector<std::vector<double>> vis(npts, std::vector<double>(3,0.0));
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);
   
    for(int i = 0; i < npts; ++i){	
	vis[i][0] = distribution(generator);
	vis[i][1] = distribution(generator);
	vis[i][2] = distribution_w(generator);
    }

    return vis;
}

static inline std::vector<double> generate_random_visibilities_(double theta,
							      double lambda,
							      double dw,
							      int npts){

    
    std::vector<double> vis(3*npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);
   
    for(int i = 0; i < npts; ++i){	
	vis[3*i + 0] = distribution(generator);
	vis[3*i + 1] = distribution(generator);
	vis[3*i + 2] = distribution_w(generator);
    }

    return vis;
}

static inline std::vector<double> generate_random_visibilities_1D_uv(double theta,
							      double lambda,
							      int npts){

    
    std::vector<double> vis(npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
   
    for(int i = 0; i < npts; ++i){	
	vis[i] = distribution(generator);
    }

    return vis;
}

static inline std::vector<double> generate_random_visibilities_1D_w(double theta,
							      double lambda,
							      double dw,
							      int npts){

    
    std::vector<double> vis(npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);
   
    for(int i = 0; i < npts; ++i){	
	vis[i] = distribution_w(generator);
    }

    return vis;
}



static inline std::vector<std::vector<double>> generate_line_visibilities(double theta,
							    double lambda,
							    double v,
							    double dw,
							      int npts){

    
    std::vector<std::vector<double>> vis(npts, std::vector<double>(3,0.0));
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);


    double npts_step = (theta*lambda)/npts;
    for(int i = 0; i < npts; ++i){	
	vis[i][0] = npts_step*i - theta*lambda/2;
	vis[i][1] = v;
	vis[i][2] = 0;
    }

    return vis;
}

static inline std::vector<double> generate_line_visibilities_(double theta,
							      double lambda,
							      double v,
							      double dw,
							      int npts){

    
    std::vector<double> vis(3*npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);

    double npts_step = (theta*lambda)/npts;
    for(int i = 0; i < npts; ++i){	
	vis[3*i + 0] = npts_step*i - theta*lambda/2;
	vis[3*i + 1] = v;
	vis[3*i + 2] = 0;
    }

    return vis;
}

#endif
