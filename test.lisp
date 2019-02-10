(ql:quickload :cl-cublas)

(in-package #:cl-cublas)

(cffi:define-foreign-library cublas
    (t (:default "libcublas")))
(cffi:use-foreign-library cublas)


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

(cublasCreate_v2 *cublas*)

(cublasAlloc 9 4 *gpuA*)
(cublasAlloc 9 4 *gpuB*)
(cublasAlloc 9 4 *gpuZ*)

(cublasSetMatrix 3 3
		 (cffi:foreign-type-size ':float)
		 *cpuA* 3
		 (cffi:mem-ref *gpuA* ':pointer) 3)

(cublasSetMatrix 3 3
		 (cffi:foreign-type-size ':float)
		 *cpuB* 3
		 (cffi:mem-ref *gpuB* ':pointer) 3)

(cublasSgemm 78 78 3 3 3 1.0
	     (cffi:mem-ref *gpuA* ':pointer) 3
	     (cffi:mem-ref *gpuB* ':pointer) 3 1.0
	     (cffi:mem-ref *gpuZ* ':pointer) 3)

(cublasGetMatrix 3 3
		 (cffi:foreign-type-size ':float)
		 (cffi:mem-ref *gpuZ* ':pointer) 3
		 *cpuZ* 3)

(print (loop for i from 0 below 8
	  collect (cffi:mem-aref *cpuZ* :float i)))
