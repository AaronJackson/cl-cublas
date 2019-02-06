CUBLAS_INCLUDE=/usr/local/cuda/include

ffi:
	swig -I${CUBLAS_INCLUDE} -cffi cublas.i
