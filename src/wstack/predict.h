#ifndef PREDICT_H
#define PREDICT_H

std::complex<double>
predict_visibility(const std::vector<double>& points,
		   double u,
		   double v,
		   double w);

std::complex<double>
predict_visibility_quantized_vec(const std::vector<double>& points,
				 double theta,
				 double lam,
				 double u,
				 double v,
				 double w);


std::vector<std::complex<double> >
predict_visibility_quantized_vec(const std::vector<double>& points,
				 double theta,
				 double lam,
				 std::vector<double> uvwvec);


// These are also used in the /test directory so declare here.

static inline std::complex<double>
deconvolve_visibility(std::vector<double> uvw,
		      double du,
		      double dw,
		      int aa_support_uv,
		      int aa_support_w,
		      int oversampling,
		      int oversampling_w,
		      int w_planes,
		      int grid_size,
		      const vector3D<std::complex<double> >& wstacks,
		      struct sep_kernel_data *grid_conv_uv,
		      struct sep_kernel_data *grid_conv_w){
    // Co-ordinates
    double u = uvw[0];
    double v = uvw[1];
    double w = uvw[2];
    
    // Begin De-convolution process using Sze-Tan Kernels.
    std::complex<double> vis_sze = {0.0,0.0};

    // U/V/W oversample values
    double flu = u - std::ceil(u/du)*du;
    double flv = v - std::ceil(v/du)*du;
    double flw = w - std::ceil(w/dw)*dw;
    
    int ovu = static_cast<int>(std::floor(std::abs(flu)/du * oversampling));
    int ovv = static_cast<int>(std::floor(std::abs(flv)/du * oversampling));
    int ovw = static_cast<int>(std::floor(std::abs(flw)/dw * oversampling_w));   
    
    int aa_h = std::floor(aa_support_uv/2);
    int aaw_h = std::floor(aa_support_w/2);

    for(int dwi = -aaw_h; dwi < aaw_h; ++dwi){
	
    	int dws = static_cast<int>(std::ceil(w/dw)) + static_cast<int>(std::floor(w_planes/2)) + dwi;
    	int aas_w = aa_support_w * ovw + (dwi+aaw_h);
    	double gridconv_w = grid_conv_w->data[aas_w];
	
    	for(int dvi = -aa_h; dvi < aa_h; ++dvi){
	    
    	    int dvs = static_cast<int>(std::ceil(v/du)) + grid_size + dvi;
    	    int aas_v = aa_support_uv * ovv + (dvi+aa_h);
    	    double gridconv_uv = gridconv_w * grid_conv_uv->data[aas_v];
	    
    	    for(int dui = -aa_h; dui < aa_h; ++dui){
		
    		int dus = static_cast<int>(std::ceil(u/du)) + grid_size + dui; 
    		int aas_u = aa_support_uv * ovu + (dui+aa_h);
    		double gridconv_u = grid_conv_uv->data[aas_u];
    		double gridconv_uvw = gridconv_uv * gridconv_u;
    		vis_sze += (wstacks(dus,dvs,dws) * gridconv_uvw );
    	    }
    	}
    }


    return vis_sze;
}


static inline std::complex<double>
deconvolve_visibility_(double u,
		       double v,
		       double w,
		       double du,
		       double dw,
		       int aa_support_uv,
		       int aa_support_w,
		       int oversampling,
		       int oversampling_w,
		       int w_planes,
		       int grid_size,
		       const vector3D<std::complex<double> >& wstacks,
		       struct sep_kernel_data *grid_conv_uv,
		       struct sep_kernel_data *grid_conv_w){
    
    // Begin De-convolution process using Sze-Tan Kernels.
    std::complex<double> vis_sze = {0.0,0.0};

    // U/V/W oversample values
    double flu = u - std::ceil(u/du)*du;
    double flv = v - std::ceil(v/du)*du;
    double flw = w - std::ceil(w/dw)*dw;
    
    // int ovu = static_cast<int>(std::floor(std::abs(flu)/du)) * oversampling;
    // int ovv = static_cast<int>(std::floor(std::abs(flv)/du)) * oversampling;
    // int ovw = static_cast<int>(std::floor(std::abs(flw)/dw)) * oversampling_w;   

    int ovu = static_cast<int>(std::floor(std::abs(flu)/du * oversampling));
    int ovv = static_cast<int>(std::floor(std::abs(flv)/du * oversampling));
    int ovw = static_cast<int>(std::floor(std::abs(flw)/dw * oversampling_w));   
    

    
    int aa_h = std::floor(aa_support_uv/2);
    int aaw_h = std::floor(aa_support_w/2);


    for(int dwi = -aaw_h; dwi < aaw_h; ++dwi){
	
	int dws = static_cast<int>(std::ceil(w/dw)) + static_cast<int>(std::floor(w_planes/2)) + dwi;
	int aas_w = aa_support_w * ovw + (dwi+aaw_h);
	double gridconv_w = grid_conv_w->data[aas_w];
	
	for(int dvi = -aa_h; dvi < aa_h; ++dvi){
	    
	    int dvs = static_cast<int>(std::ceil(v/du)) + grid_size + dvi;
	    int aas_v = aa_support_uv * ovv + (dvi+aa_h);
	    double gridconv_uv = gridconv_w * grid_conv_uv->data[aas_v];
	    
	    for(int dui = -aa_h; dui < aa_h; ++dui){
		
		int dus = static_cast<int>(std::ceil(u/du)) + grid_size + dui; 
		int aas_u = aa_support_uv * ovu + (dui+aa_h);
		double gridconv_u = grid_conv_uv->data[aas_u];
		double gridconv_uvw = gridconv_uv * gridconv_u;
		vis_sze += (wstacks(dus,dvs,dws) * gridconv_uvw );
	    }
	}
    }

    return vis_sze;
}

std::vector<std::complex<double> >
wstack_predict(double theta,
	       double lam,
	       const std::vector<double>& points,
	       const std::vector<double> uvwvec,
	       double du, // Sze-Tan Optimum Spacing in U/V
	       double dw, // Sze-Tan Optimum Spacing in W
	       int aa_support_uv,
	       int aa_support_w,
	       double x0,
	       struct sep_kernel_data *grid_conv_uv,
	       struct sep_kernel_data *grid_conv_w,
	       struct sep_kernel_data *grid_corr_lm,
	       struct sep_kernel_data *grid_corr_n);

std::vector<std::vector<std::complex<double>>>
wstack_predict_lines(double theta,
		     double lam,
		     const std::vector<double>& points, // Sky points
		     const std::vector<std::vector<double>> uvwvec, // U/V/W points to predict.
		     double du, // Sze-Tan Optimum Spacing in U/V
		     double dw, // Sze-Tan Optimum Spacing in W
		     int aa_support_uv,
		     int aa_support_w,
		     double x0,
		     struct sep_kernel_data *grid_conv_uv,
		     struct sep_kernel_data *grid_conv_w,
		     struct sep_kernel_data *grid_corr_lm,
		     struct sep_kernel_data *grid_corr_n);



#endif
