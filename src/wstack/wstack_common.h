#include "../common/hdf5.cuh"

#ifndef WSTACK_H
#define WSTACK_H

#define THREADS_BLOCK 16

#include <iostream>
#include <vector>
#include <complex>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstring>



/*********************************************************
        Stride Constants - *IMPORTANT*

  Using non-unit strides will alleviate cache thrashing
  effects. One of the most powerful ways to increase
  (or decrease!) performance.

**********************************************************/
const int element_stride = 1;
const int row_stride = 8;
const int matrix_stride = 10;
/*********************************************************/



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
       T = datatype
       s1 = Distance (in terms of units of T) in between contiguous array elements
       s2 = Offset (in terms of units of T) between row elements.
       s3 = Offset (in terms of units of T) between matrix elements.

       I decided against byte striding because it can decrease performance.

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
    // 3D Access
    T & operator()(size_t i, size_t j, size_t k) {
	//return data[k*d1*d2 + j*d1 + i];
        return data[k*n3s + j*n2s + i*s1];
    }

    T const & operator()(size_t i, size_t j, size_t k) const {
	//return data[k*d1*d2 + j*d1 + i];
        return data[k*n3s + j*n2s + i*s1];
    }

    T & operator()(size_t i, size_t j) {
	//return data[j*d1 + i];
	return data[j*(d1*s1+s2) + i*s1];
    }
    
    // 2D Access
    T const & operator()(size_t i, size_t j) const {
	//return data[j*d1 + i];
	return data[j*(d1*s1+s2) + i*s1];
    }
    
    T & operator()(size_t i) {
        return data[i];
    }
    // 1D Access
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
	assert(planei < d3);
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


// Stealing Peters code has become the hallmark of my PhD.
template <typename T>
void fft_shift_2Darray(vector3D<T>& array,
		       std::size_t planei){

    std::size_t grid_sizex = array.d1s();
    std::size_t grid_sizey = array.d2s();
    
    assert(grid_sizex % 2 == 0);
    assert(grid_sizey % 2 == 0);
    int i1,j1;
    for (std::size_t j = 0; j < grid_sizex; ++j){
	for (std::size_t i = 0; i < grid_sizey/2; ++i){
	    // int ix0 = j * grid_sizex + i;
	    // int ix1 = (ix0 + (grid_sizex + 1) * (grid_sizex/2)) % (grid_sizex * grid_sizey);

	    i1 = i + grid_sizex/2;
	    if (j < grid_sizey/2){
		j1 = j + grid_sizey/2;
	    } else {
		j1 = j - grid_sizey/2;
	    }
	   
	    T temp = array(i,j,planei);
	    array(i,j,planei) = array(i1,j1,planei);
	    array(i1,j1,planei) = temp;
	}
    }
}



// Stealing Peters code has become the hallmark of my PhD.
template <typename T>
void fft_shift_2Darray(vector2D<T>& array){

    std::size_t grid_sizex = array.d1s();
    std::size_t grid_sizey = array.d2s();
    
    assert(grid_sizex % 2 == 0);
    assert(grid_sizey % 2 == 0);
    int i1,j1;
    for (std::size_t j = 0; j < grid_sizex; ++j){
	for (std::size_t i = 0; i < grid_sizey/2; ++i){
	    // int ix0 = j * grid_sizex + i;
	    // int ix1 = (ix0 + (grid_sizex + 1) * (grid_sizex/2)) % (grid_sizex * grid_sizey);

	    i1 = i + grid_sizex/2;
	    if (j < grid_sizey/2){
		j1 = j + grid_sizey/2;
	    } else {
		j1 = j - grid_sizey/2;
	    }
	   
	    T temp = array(i,j);
	    array(i,j) = array(i1,j1);
	    array(i1,j1) = temp;
	}
    }
}

template <typename T>
void zero_pad_2Darray(const vector3D<T>& array,
		      vector2D<T>& padded_array,
		      double x0,
		      std::size_t planei){

    int x0i = static_cast<int>(std::round(1.0/x0));
    std::cout << "xoi: " << x0i << "\n";
    int i0 = padded_array.d1s()/x0i;
    int i1 = 3*(padded_array.d1s()/x0i);
    int j0 = padded_array.d2s()/x0i;
    int j1 = 3*(padded_array.d2s()/x0i);
    for(int j = j0; j < j1; ++j){
	for(int i = i0; i < i1; ++i){
	    padded_array(i,j) = array(i-i0,j-j0,planei);
	}
    }
}

template <typename T>
void zero_pad_2Darray(const vector2D<T>& array,
		      vector2D<T>& padded_array,
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

template <typename T>
void memcpy_plane_to_stack(vector2D<T>&plane,
			   vector3D<T>&stacks,
			   std::size_t grid_size,
			   std::size_t planei,
			   std::size_t direction){

    //Calculate memory copy amount based on striding information.
    //Assume strides for n=1 and n=2 dimensions are the same between
    //stacks and plane. If they aren't then can't use memcpy directly.

    std::size_t p1s,p2s,s1s,s2s,s3s,p1d,p2d,s1d,s2d;


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
    s3s = stacks.s3s();
    
    // Let us really make sure the strides are the same
    assert(p1s == s1s);
    assert(p2s = s2s);

    

    std::size_t copy_size = (p1d*p1s + p2s) * p2d * sizeof(T); 

    T *wp = stacks.pp(planei);  
    T *pp = plane.dp();

    if (direction){
	std::memcpy(wp,pp,copy_size);
    } else {
	std::memcpy(pp,wp,copy_size);
    }

}



void multiply_fresnel_pattern(vector2D<std::complex<double> >& fresnel,
			      vector3D<std::complex<double> >& sky,
			      int t,
			      std::size_t planei);

void multiply_fresnel_pattern(vector2D<std::complex<double> >& fresnel,
			      vector2D<std::complex<double> >& sky,
			      int t);

void memcpy_plane_to_stack(vector2D<std::complex<double> >&plane,
			   vector3D<std::complex<double> >&stacks,
			   std::size_t grid_size,
			   std::size_t planei,
			   std::size_t direction = 1);

vector2D<std::complex<double> >
generate_fresnel(double theta,
		 double lam,
		 double dw,
		 double x0);

//Quantises our sources onto the sky.
void generate_sky(const std::vector<double>& points,
		  vector2D<std::complex<double> >& sky, // We do this by reference because of FFTW planner.
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

// Various inlines for generating visibilities, mostly useful for /test

static inline std::vector<std::vector<double> > generate_random_visibilities(double theta,
							      double lambda,
							      double dw,
							      int npts){

    
    std::vector<std::vector<double> > vis(npts, std::vector<double>(3,0.0));
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

static inline std::vector<double> generate_random_visibilities_1D_uv_gaussian(double theta,
								   double lambda,
									    int npts){

    std::vector<double> vis(npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::normal_distribution<double> distribution(0,theta*lambda/8);
    for(int i = 0; i < npts; ++i){
	vis[i] = distribution(generator);
	if((vis[i] < -theta*lambda/2) || (vis[i] >= theta*lambda/2)) --i;
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

static inline std::vector<double> generate_random_visibilities_1D_w_gaussian(double theta,
									     double lambda,
									     double dw,
									     int npts){
    std::vector<double> vis(npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::normal_distribution<double> distribution_w(0,dw);
    for(int i = 0; i < npts; ++i){	
	vis[i] = distribution_w(generator);
	if((vis[i] < -2*dw) || (vis[i] > 2*dw)) --i; 
    }
    
    return vis;
}



static inline std::vector<std::vector<double> > generate_line_visibilities(double theta,
							    double lambda,
							    double v,
							    double dw,
							      int npts){

    
    std::vector<std::vector<double> > vis(npts, std::vector<double>(3,0.0));
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
