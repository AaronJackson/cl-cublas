(load "~/.sbclrc")
(ql:quickload :cffi)
(ql:quickload :cffi-libffi)

(cffi:define-foreign-library cublas
    (t (:default "libcublas")))
(cffi:use-foreign-library cublas)

(load "cublas.lisp")

(defun cublasCreate (pointer)
  (cffi:foreign-funcall "cublasCreate_v2"
			:pointer pointer
			cublasStatus_t))

(defun cublasShutdown (pointer)
  (cffi:foreign-funcall "cublasShutdown"
			:pointer pointer
			cublasStatus_t))

(defun cublasAlloc (length size ptr)
  (cffi:foreign-funcall "cublasAlloc"
			:int length
			:int size
			:pointer ptr
			cublasStatus_t))

(defun cublasSetMatrix (rows cols size src dst)
  (cffi:foreign-funcall "cublasSetMatrix"
			:int rows
			:int cols
			:int size
			:pointer src ; cpu tensor
			:int rows
			:pointer dst ; gpu tensor
			:int cols
			cublasStatus_t))

(defun cublasGetMatrix (rows cols size src dst)
  (cffi:foreign-funcall "cublasGetMatrix"
			:int rows
			:int cols
			:int size
			:pointer src ; gpu tensor
			:int rows
			:pointer dst ; cpu tensor
			:int cols
			cublasStatus_t))

(defun sgemm (rowsA colsB colsA alpha A lda B ldb beta C ldc)
  (cffi:foreign-funcall "cublasSgemm"
			:char 78 ;; transa (row first)
			:char 78 ;; transb (row first)
			:int rowsA :int colsB :int colsA
			:float alpha
			:pointer A
			:int lda
			:pointer B
			:int ldb
			:float beta
			:pointer C
			:int ldc
			cublasStatus_t))

(defparameter *cublas* (cffi:foreign-alloc :pointer))
(defparameter *cpuA* (cffi:foreign-alloc
		      :float
		      :initial-contents #(2.0 2.0 2.0
					  2.0 2.0 2.0
					  2.0 2.0 2.0)))
(defparameter *cpuB* (cffi:foreign-alloc
		      :float
		      :initial-contents #(1.0 2.0 3.0
					  4.0 5.0 6.0
					  7.0 8.0 9.0)))
(defparameter *gpuA* (cffi:foreign-alloc :pointer))
(defparameter *gpuB* (cffi:foreign-alloc :pointer))
(defparameter *cpuZ* (cffi:foreign-alloc
		      :float :count 9 :initial-element 0.0))
(defparameter *gpuZ* (cffi:foreign-alloc :pointer))

(if (= (cffi:foreign-enum-value 'cublasStatus_t (cublasCreate *cublas*))
       (cffi:foreign-enum-value 'cublasStatus_t :CUBLAS_STATUS_SUCCESS))
    (print "cuBLAS initialised successfully."))

(print (cublasAlloc 9 4 *gpuA*))
(print (cublasAlloc 9 4 *gpuB*))
(print (cublasAlloc 9 4 *gpuZ*))

(print (cublasSetMatrix 3 3
			(cffi:foreign-type-size ':float)
			*cpuA*
			(cffi:mem-ref *gpuA* ':pointer)))
(print (cublasSetMatrix 3 3
			(cffi:foreign-type-size ':float)
			*cpuB*
			(cffi:mem-ref *gpuB* ':pointer)))

(print (sgemm 3 3 3 1.0
	      (cffi:mem-ref *gpuA* ':pointer) 3
	      (cffi:mem-ref *gpuB* ':pointer) 3 1.0
	      (cffi:mem-ref *gpuZ* ':pointer) 3))

(print (cublasGetMatrix 3 3
			(cffi:foreign-type-size ':float)
			(cffi:mem-ref *gpuZ* ':pointer)
			*cpuZ*))

(print (loop for i from 0 below 8
	  collect (cffi:mem-aref *cpuZ* :float i)))

(if (= (cffi:foreign-enum-value 'cublasStatus_t (cublasShutdown *cublas*))
       (cffi:foreign-enum-value 'cublasStatus_t :CUBLAS_STATUS_SUCCESS))
    (print "cuBLAS shutdown successfully."))
