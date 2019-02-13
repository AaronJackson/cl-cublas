(defpackage #:cl-cublas
  (:use #:cl #:cffi))

(cffi:define-foreign-library cublas
    (t (:default "libcublas")))
(cffi:use-foreign-library cublas)
