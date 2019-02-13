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
   (ptr-cpu
    :accessor ptr-cpu
    :initform nil)
   (ptr-gpu
    :accessor ptr-gpu
    :initform nil)
   (current-ptr ;; specifies whether the latest copy is on cpu or gpu
    :accessor current-ptr
    :initform 'cpu)))

(defmethod set-data ((m matrix) data)
  "set data of matrix"
  (setf (ptr-cpu m) (cffi:foreign-alloc
		     :float :count (* (rows m) (cols m))
		     :initial-element 0.0))
  (dotimes (r (rows m))
    (dotimes (c (cols m))
      (setf (cffi:mem-aref (ptr-cpu m) :float (+ (* (cols m) r) c))
	    (aref data (+ (* (rows m) c) r)))))
  (setf (current-ptr m) 'cpu)
  m)

(defmethod gpu ((m matrix))
  "push c data to cuda data"
  (if (eq (current-ptr m) 'cpu)
      (progn (if (not (ptr-gpu m))
		 (progn (setf (ptr-gpu m)
			      (cffi:foreign-alloc :pointer))
			(cublasAlloc (* (rows m) (cols m))
				     (cffi:foreign-type-size ':float)
				     (ptr-gpu m))))
	     (cublasSetMatrix (rows m) (cols m)
			      (cffi:foreign-type-size ':float)
			      (ptr-cpu m) (rows m)
			      (cffi:mem-ref (ptr-gpu m) ':pointer) (cols m))))
  (setf (current-ptr m) 'gpu)
  m)

(defmethod cpu ((m matrix))
  "pull cuda data to c data"
  (if (eq (current-ptr m) 'gpu)
      (cublasGetMatrix (rows m) (cols m)
		       (cffi:foreign-type-size ':float)
		       (cffi:mem-ref (ptr-gpu m) ':pointer)
		       (rows m) (ptr-cpu m) (cols m)))
  (setf (current-ptr m) 'cpu)
  m)

(defmethod print-object ((m matrix) stream)
  "prints a matrix, uh, as a list for now"
  (cpu m)
  (dotimes (c (cols m))
       (dotimes (r (rows m))
	 (format stream "~$ " (cffi:mem-aref (ptr-cpu m) :float
					     (+ (* (cols m) r) c))))
       (format stream "~%"))
  m)

;; The dimensions may be wrong, haven't really thought about them much
;; since I'm only trying a 3x3 matrix at the moment.
(defmethod multiply ((A matrix) (B matrix))
  "A"
  (let ((Z (make-instance 'matrix :rows (rows A) :cols (cols B))))
    (set-data Z (make-array (* (rows A) (cols A))
			    :element-type ':float
			    :initial-element 0.0))
    (cublasSgemm 78 78 (cols A) (rows B) (rows A) 1.0
		 (cffi:mem-ref (ptr-gpu (gpu A)) ':pointer) (rows A)
		 (cffi:mem-ref (ptr-gpu (gpu B)) ':pointer) (rows A) 1.0
		 (cffi:mem-ref (ptr-gpu (gpu Z)) ':pointer) (cols A))
    Z))

(defvar A (make-instance 'matrix :rows 3 :cols 3))
(defvar B (make-instance 'matrix :rows 3 :cols 3))

(set-data A #(1.0 2.0 3.0
	      4.0 5.0 6.0
	      7.0 8.0 9.0))
(set-data B #(1.0 2.0 1.0
	      2.0 1.0 2.0
	      1.0 2.0 1.0))

(print A)
(print (multiply A B))

;; Should print
;;      8    10     8
;;     20    25    20
;;     32    40    32

