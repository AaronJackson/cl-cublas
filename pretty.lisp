(load "~/.sbclrc")
(ql:quickload :cffi)
(ql:quickload :cffi-libffi)

(cffi:define-foreign-library cublas
    (t (:default "libcublas")))
(cffi:use-foreign-library cublas)

(load "cublas.lisp")

(defclass matrix ()
  ((rows
    :initarg :rows
    :accessor rows)
   (cols
    :initarg :cols
    :accessor cols)
   (ptr-c
    :accessor ptr-c
    :initform nil)
   (ptr-gpu
    :accessor ptr-gpu
    :initform nil)))

(defmethod set-matrix ((m matrix) data)
  "set data of matrix"
  (setf (slot-value m 'ptr-c)
	(cffi:foreign-alloc
	 :float
	 :initial-contents data)))

(defmethod push-matrix (m)
  "push c data to cuda data"
  (if (not (ptr-gpu m))
      (setf
	  (slot-value m 'ptr-gpu)
	  (cffi:foreign-alloc :pointer))
      (cublasAlloc
       (* (rows m) (cols m)) 4
       (slot-value m 'ptr-gpu)))
  (cublasSetMatrix (rows m) (cols m)
		 (cffi:foreign-type-size ':float)
		 (ptr-c m) (rows m)
		 (cffi:mem-ref (slot-value m 'ptr-gpu) ':pointer) (cols m)))

(defmethod pull-matrix (m)
  "pull cuda data to c data"
  (cublasGetMatrix (rows m) (cols m)
		 (cffi:foreign-type-size ':float)
		 (cffi:mem-ref (ptr-gpu m) ':pointer)
		 (rows m) (slot-value m 'ptr-c) (cols m)))

(defmethod print-matrix (m)
  "prints a matrix, uh, as a list for now"
  (print (loop for i from 0 below (* (rows m) (cols m))
	    collect (cffi:mem-aref (ptr-c m) :float i)))
  T)

;; (defmethod multiply-matrix ((A matrix) (B matrix))
;;   "A * B"
;;   (let (Z (make-instance 'matrix :rows (rows A) :cols (cols B)))
;;     (set-matrix Z (make-array
;; 		   (* (rows A) (cols B))
;; 		   :initial-element 0.0))
;;     (push-matrix Z)
;;     (cublasSgemm 78 78 (cols A) (rows B) (rows A) 1.0
;; 		 (cffi:mem-ref (ptr-gpu A) ':pointer) (rows A)
;; 		 (cffi:mem-ref (ptr-gpu B) ':pointer) (cols A) 1.0
;; 		 (cffi:mem-ref (slot-value Z 'ptr-gpu) ':pointer)))
;;   Z)



(defvar A (make-instance 'matrix :rows 3 :cols 3))
(defvar B (make-instance 'matrix :rows 3 :cols 3))

(set-matrix A #(2.0 2.0 2.0
		2.0 2.0 2.0
		2.0 2.0 2.0))
(set-matrix B #(1.0 0.0 0.0
		0.0 1.0 0.0
		0.0 0.0 1.0))
(push-matrix A)
(pull-matrix A)
(print-matrix A)

; This bit does not work yet.
;(defvar C (multiply-matrix A B))
;(pull-matrix C)
;(print-matrix C)
