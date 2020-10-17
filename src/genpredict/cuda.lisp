;; Copyright James Kent, 2020

;; This file sets up the boilerplate for the predict kernels in
;; NVIDIA CUDA.

(defun cuda-interface (kernel-name template-type complex-type)
  (format nil
	  "template <typename ~a> \
__global__ void ~a(~a* __restrict__ wstacks, \
                   ~a* vis, \
                   ~a* __restrict__ uvec, \
                   ~a* __restrict__ vvec, \
                   ~a* __restrict__ wvec, \
                   ~a* __restrict__ gcf_uv, \
                   ~a* __restrict__ gcf_w, \
                   ~a du, \
                   ~a dw, \
                   int vis_nu, \
		   int aa_support_uv, \
		   int aa_support_w, \
		   int oversampling, \
		   int oversampling_w, \
		   int w_planes, \
		   int grid_size, \
		   int oversampg){ \
    
    
    const unsigned int aa_h = aa_support_uv/2; \
    const unsigned int aaw_h = aa_support_w/2; \
    \
    // Warp Information \
    \
    const unsigned int warp_total = blockDim.x / 32; \
    const unsigned int warp_idx = threadIdx.x / 32; \
    const unsigned int lane_idx = threadIdx.x % 32; \
    const unsigned int quarter_warp_idx = lane_idx / 8; \
    const unsigned int quarter_warp_lane_idx = lane_idx % 8; \
    // Shared memory \ 
    extern __shared__ unsigned int array[]; \

    // Offset for each warp in the u/v/w vectors \
    const unsigned int uvw_offset = ILP*(blockIdx.x * warp_total + warp_idx); \
    // Start pointer for the saved visibility values \
    thrust::complex<FloatType> *vis_space = \
	reinterpret_cast<thrust::complex<FloatType>*>(array) + warp_idx * 32; \ "
	  (string template-type)
	  (string kernel-name)
	  (string complex-type)
	  (string complex-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)))

(defun cuda-loop-start (ilp-var template-type)
  (format nil
	  "for(int i = 0; i < ~a; ++i){\
	~a u = uvec[uvw_offset + i];\
	~a u = vvec[uvw_offset + i];\
	~a u = wvec[uvw_offset + i];\
\
	~a flu = u - cuda_ceil(u/du)*du;\
	~a flv = v - cuda_ceil(v/du)*du;\
	~a flw = w - cuda_ceil(w/dw)*dw;\
\
	int ovu = static_cast<int>(cuda_floor(cuda_fabs(flu)/du * oversampling)):\
	int ovv = static_cast<int>(cuda_floor(cuda_fabs(flv)/du * oversampling)):\
	int ovw = static_cast<int>(cuda_floor(cuda_fabs(flw)/dw * oversampling_w)):\ "
	  (string ilp-var)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)
	  (string template-type)))


(defun cuda-loop-end (shared-mem)
  (format nil
"}\
__syncwarp();\
vis[uvw_offset + lane_idx] = vis_space[lane_idx];"))

(defun cuda-warp-shuffle (template-type complex-type sum-variable shared-mem)
  (format nil
"		~a realn = ~a.real(); \
		~a imagn = ~a.imag(); \
		__syncwarp(); // Synchronise warp before reduction \ 
\

		// Warp shuffle reduction \
		for(int offset = 16; offset > 0; offset /= 2){
			realn += __shfl_down_sync(0xFFFFFFFF,realn,offset); \
			imagn += __shfl_down_sync(0xFFFFFFFF,imagn,offset); \
		} \
		if(lane_idx == 0){ \
			~a temp = {realn, imagn}; \
			~a[i] = temp; \
	  	}"
	  (string template-type)
	  (string sum-variable)
	  (string template-type)
	  (string sum-variable)
	  (string complex-type)
	  (string shared-mem)))


(format t (cuda-warp-shuffle "FloatType" "thrust::complex<FloatType>" "value" "vis_space"))
