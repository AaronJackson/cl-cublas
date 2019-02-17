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

;; Matrices can be transposed, but tranposing a matrix simply sets a
;; flag. cuBLAS handles the rest, making transpositions very fast.
(let ((A (set-data (matrix 2 2) #(2.0 1.0
				  2.0 2.0)))
      (B (set-data (matrix 2 2) #(2.0 2.0
				  1.0 1.0))))
  (print (multiply A B))
  (transpose B)
  (print (multiply A B))
  (transpose B)
  (print (multiply A B)))

;; Checking non-square matrix transpose
(let ((A (ones 2 3))
      (B (ones 2 3))
      (Z (zeros 2 2)))
  (transpose B)
  (multiply-to A B Z)
  (print Z))

;; The print matrix function is aware of this tranpose flag.
(let ((A (rand 4 3)))
  (print A)
  (transpose A)
  (print A))
