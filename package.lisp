(defpackage #:cl-cublas
  (:use #:cl #:cffi)
  (:export #:print-object
	   #:zeros
	   #:ones
	   #:eye
	   #:transpose
	   #:multiply-to
	   #:multiply
	   #:add
	   #:rand
	   #:randn))

(cffi:define-foreign-library cublas
    (t (:default "libcublas")))
(cffi:use-foreign-library cublas)
