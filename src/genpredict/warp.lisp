;;; DEFAULTS - Do not change these unless you really know
;;; what you are doing!

;; Most NVIDIA/Apple GPU's have warp size of 32. AMD uses 64
;; in some recent cards. Critical this is set properly for
;; the fastest (and correct) results.

;; This inner auto-coalesced inner loop should
;; be agnostic to implementation provided it's implemented in
;; some C-Style such as C(OpenCL)/C++ (CUDA)/Objective-C (Apple Metal).
;; We assume the implementation language is ANSI C-compliant.

(defvar warp-size 32)
(defvar max-convolution-size warp-size)

;; TODO: Should move these somewhere else
(defvar complex-type "thrust::complex<FloatType>")
(defvar float-type "double")
(defvar wstack-var "wstacks")
(defvar accum-val "value")
(defvar gcf-uv-var "gcf_uv")
(defvar gcf-w-var "gcf_w")
(defvar aa-coord-u "aas_u")
(defvar aa-coord-v "aas_v")
(defvar aa-coord-w "aas_w")

(defun range (max &key (min 0) (step 1))
   (loop for n from min below max by step
	 collect n))

(defun v-loop (cast-style floor-fn v-size)
  (format nil
	  "for(unsigned int lb_v = 0; lb_v < ~a; ++lb_v){\
unsigned int v_grid = ~a(~a(v/du)) + grid_size - aa_h + lb_v;\
int aas_v = aa_support_uv * ovv + lb_v;"
	  (string v-size)
	  (string cast-style)
	  (string floor-fn)))


(defun convolve-accumulate (complex-type
			    wstack-var
			    float-type
			    accum-val
			    gcf-w-var
			    gcf-uv-var
			    aa-coord-u
			    aa-coord-v
			    aa-coord-w)
  (format nil "~a grid = ~a[grid_coord + v_grid * oversampg];\
~a conv_value = 1.0 * ~a[~a] * ~a[~a] * ~a[~a];\
~a \+= grid * conv_value;~%"
	  (string complex-type)
	  (string wstack-var)
	  (string float-type)
	  (string gcf-w-var)
	  (string aa-coord-w)
	  (string gcf-uv-var)
	  (string aa-coord-u)
	  (string gcf-uv-var)
	  (string aa-coord-v)
	  (string accum-val)))

(defun kernel-offsets (support-var-uv
		       support-var-w
		       quarter-warp-idx-var
		       quarter-warp-lane-idx-var
		       oversampling-var-uv
		       oversampling-var-w
		       aas-uv-var
		       aas-w-var)
  (format nil
	  "int ~a = ~a * ~a + ~a;\
int ~a = ~a * ~a + ~a;"
	  (string aas-uv-var)
	  (string support-var-uv)
	  (string oversampling-var-uv)
	  (string quarter-warp-idx-var)
	  (string aas-w-var)
	  (string support-var-w)
	  (string oversampling-var-w)
	  (string quater-warp-idx-var)))

(defun generate-predict (u-size v-size w-size)
  (let ((syntax-list nil)
	(subwarp-size u-size)
	(subwarps (/ warp-size u-size))
	(w-delta (/ w-size (/ warp-size u-size))))
    (if (> w-delta 1)
	(progn
	  (if (>= w-delta 2)
	      (setq syntax-list (append syntax-list (list
			    (format nil "for(unsigned int lb_w = 0; lb_w < ~a; ++lb_w){" (string (digit-char w-delta)))))))
	  (setq syntax-list (append syntax-list (list (v-loop "static_cast<int>" "cuda_ceil" (digit-char v-size)))))
	  (setq syntax-list (append syntax-list (list (convolve-accumulate
				     complex-type
				     wstack-var
				     float-type
				     accum-val
				     gcf-w-var
				     gcf-uv-var
				     aa-coord-u
				     aa-coord-v
				     aa-coord-w))))
	  )
	(error "The u/v/w sizing does not work for the subwarp scheme here. Hand optimise perhaps?"))))
	
		  


(generate-predict 8 8 8)


