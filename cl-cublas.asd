;;;; cl-cublas.asd

(defsystem #:cl-cublas
  :description "Playing about with cuBLAS in Common Lisp"
  :author "Aaron Jackson <aaron.jackson@nottingham.ac.uk>"
  :depends-on (#:cffi
	       #:cffi-libffi
	       #:trivial-garbage)
  :serial t
  :components ((:file "package")
               (:file "cublas")))

