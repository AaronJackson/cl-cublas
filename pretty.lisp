(push '*default-pathname-defaults* asdf:*central-registry*)
(asdf:load-system :cl-cublas)

(in-package :cl-cublas)

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
   (current-ptr
    ;; Specifies whether the "most up-to-date" copy of data is in host
    ;; or device memory. It should be set to either 'cpu or 'gpu, and
    ;; this should be automatic.
    :accessor current-ptr
    :initform 'cpu)))

(defmethod set-data ((m matrix) data)
  "set data of matrix"
  (setf (ptr-cpu m) (cffi:foreign-alloc
		     :float :count (* (rows m) (cols m))
		     :initial-element 0.0))
  (dotimes (r (rows m))
    (dotimes (c (cols m))
      (setf (cffi:mem-aref (ptr-cpu m) :float (+ (* (rows m) c) r))
	    (aref data (+ (* (cols m) r) c)))))
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
	     (assert (eq :CUBLAS_STATUS_SUCCESS
			 (cublasSetMatrix (rows m) (cols m)
					  (cffi:foreign-type-size ':float)
					  (ptr-cpu m) (rows m)
					  (cffi:mem-ref (ptr-gpu m) ':pointer)
					  (rows m))))))
  (setf (current-ptr m) 'gpu)
  m)

(defmethod cpu ((m matrix))
  "pull cuda data to c data"
  (if (eq (current-ptr m) 'gpu)
      (assert (eq :CUBLAS_STATUS_SUCCESS
		  (cublasGetMatrix (rows m) (cols m)
				   (cffi:foreign-type-size ':float)
				   (cffi:mem-ref (ptr-gpu m) ':pointer)
				   (rows m) (ptr-cpu m) (rows m)))))
  (setf (current-ptr m) 'cpu)
  m)

(defmethod cleanup ((m matrix))
  (cublasFree (ptr-gpu m))
  (cffi:foreign-free (ptr-cpu m))
  (setf (rows m) 0
	(cols m) 0)
  nil)


(defmethod print-object ((m matrix) stream)
  "prints a matrix, uh, as a list for now"
  (cpu m)
  (dotimes (r (rows m))
    (dotimes (c (cols m))
      (format stream "~2,1,6$ " (cffi:mem-aref (ptr-cpu m) :float
					  (+ (* (rows m) c) r))))
    (format stream "~%"))
  m)

(defmethod zeros (r c)
  (let ((Z (make-instance 'matrix :rows r :cols c)))
    (set-data Z (make-array (* r c) :initial-element 0.0))
    Z))

(defmethod ones (r c)
  (let ((Z (make-instance 'matrix :rows r :cols c)))
    (set-data Z (make-array (* r c) :initial-element 1.0))
    Z))

(defmethod eye (r)
  "Returns the identity matrix of size r x r"
  (let ((Z (make-instance 'matrix :rows r :cols r)))
    (setf (ptr-cpu Z) (cffi:foreign-alloc :float
					  :count (* r r)
					  :initial-element 0.0))
    (dotimes (i r)
      (setf (cffi:mem-aref (ptr-cpu Z) :float (+ (* r i) i)) 1.0))
  Z))

(defmethod multiply-to ((A matrix) (B matrix) (Z matrix))
  "Multiply matrices A and B, storing result in Z (returned)"
  ;; Ensure that the inner dimensions of A and B match.
  (assert (= (cols A) (rows B)))
  ;; Ensure that the outer dimensions of A and B match the size of Z.
  (assert (and (= (rows A) (rows Z)) (= (cols B) (cols Z))))
  (let ((m (rows A)) (n (cols B)) (k (cols A)) (alpha 1.0) (beta 0.0))
    ;; 78 is ASCII N, i.e. no transpose.
    (cublasSgemm 78 78 m n k alpha
		 (cffi:mem-ref (ptr-gpu (gpu A)) ':pointer) m
		 (cffi:mem-ref (ptr-gpu (gpu B)) ':pointer) k beta
		 (cffi:mem-ref (ptr-gpu (gpu Z)) ':pointer) m))
  Z)

(defmethod multiply ((A matrix) (B matrix))
  "Multiply matrices A and B, preallocating returned Z"
  (assert (= (cols A) (rows B)))
  (let ((Z (zeros (rows A) (cols B))))
    (multiply-to A B Z)
    Z))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Testing stuff:

(let ((a (ones 1024 10000))
      (b (ones 10000 1024))
      (z (zeros 1024 1024)))
  (dotimes (i 1)
    (multiply-to a b z))
  (cleanup a)
  (cleanup b)
  (cleanup z))
