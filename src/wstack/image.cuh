#ifndef IMAGE_CUH
#define IMAGE_CUH

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
		struct sep_kernel_data *grid_corr_n);


#endif