(push '*default-pathname-defaults* asdf:*central-registry*)
(asdf:load-system :cl-cublas)

(in-package :cl-cublas)

(cffi:define-foreign-library cublas
    (t (:default "libcublas")))
(cffi:use-foreign-library cublas)

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

(defmethod set-data ((m matrix) data)
  "set data of matrix"
  (setf (slot-value m 'ptr-c)
	(cffi:foreign-alloc
	 :float
	 :initial-contents data))
  m)

(defmethod gpu ((m matrix))
  "push c data to cuda data"
  (if (not (ptr-gpu m))
      (progn (setf (ptr-gpu m)
		   (cffi:foreign-alloc :pointer))
	     (cublasAlloc (* (rows m) (cols m))
			  (cffi:foreign-type-size ':float)
			  (ptr-gpu m))))
  (cublasSetMatrix (rows m) (cols m)
		   (cffi:foreign-type-size ':float)
		   (ptr-c m) (rows m)
		   (cffi:mem-ref (ptr-gpu m) ':pointer) (cols m))
  m)

(defmethod cpu ((m matrix))
  "pull cuda data to c data"
  (cublasGetMatrix (rows m) (cols m)
		   (cffi:foreign-type-size ':float)
		   (cffi:mem-ref (ptr-gpu m) ':pointer)
		   (rows m) (ptr-c m) (cols m))
  m)

(defmethod print-object ((m matrix) stream)
  "prints a matrix, uh, as a list for now"
  (format stream "~$" (loop for i from 0 below (* (rows m) (cols m))
		    collect (cffi:mem-aref (ptr-c m) :float i)))
  m)

;; The dimensions may be wrong, haven't really thought about them much
;; since I'm only trying a 3x3 matrix at the moment.
(defmethod multiply ((A matrix) (B matrix))
  "A"
  (let ((Z (make-instance 'matrix :rows (rows A) :cols (cols B))))
    (set-data Z (make-array (* (rows A) (cols A))
			      :initial-element 0.0))
    (gpu Z)
    (cublasSgemm 78 78 (cols A) (rows B) (rows A) 1.0
		 (cffi:mem-ref (ptr-gpu A) ':pointer) (rows A)
		 (cffi:mem-ref (ptr-gpu B) ':pointer) (rows A) 1.0
		 (cffi:mem-ref (ptr-gpu Z) ':pointer) (cols A))
    Z))

(defvar A (make-instance 'matrix :rows 3 :cols 3))
(defvar B (make-instance 'matrix :rows 3 :cols 3))

(gpu (set-data A #(1.0 2.0 3.0
		   4.0 5.0 6.0
		   7.0 8.0 9.0)))
(gpu (set-data B #(1.0 2.0 1.0
		   2.0 1.0 2.0
		   1.0 2.0 1.0)))

;; Hmm output is transposed. Might need to switch to column first
;; indexing.
(print (cpu (multiply (gpu A) (gpu B))))
