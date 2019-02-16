;; Feel free to remove this line if you have cloned the project to
;; your quicklisp local-projects directory.
(push '*default-pathname-defaults* asdf:*central-registry*)

(asdf:load-system :cl-cublas)
(in-package #:cl-cublas)

;; Print a simple 3x3 matrix of ones.
(print (ones 3 3))

;; Do a simple multiplication against an identity
(print (multiply (ones 3 3) (eye 3)))

;; While the multiplication was done on the GPU, allocation of GPU
;; memory and copying the data will have taken some time. You can make
;; this much faster by keeping your data on the gpu.
(let ((A (ones 4 1024))
      (B (ones 1024 4))
      (Z (zeros 4 4)))
  (time (dotimes (i 1000)
	  (multiply-to A B Z))))

;; Several other convenience functions are available, such as rand and
;; randn.
(print (rand 3 3)) ;; uniform
(print (randn 3 3)) ;; normal (estimated)
