(in-package :cl-cublas)

(defclass <matrix> ()
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

(defun attach-cpu-finalizer (m)
  (let ((p (ptr-cpu m)))
    (trivial-garbage:finalize m (lambda ()
				  (cffi:foreign-free p)))))

(defun attach-gpu-finalizer (m)
  (let ((p (ptr-gpu m)))
    (trivial-garbage:finalize m (lambda ()
				  (cublasFree p)))))

(defun matrix (rows cols)
  (make-instance '<matrix> :rows rows :cols cols))

(defmethod set-data ((m <matrix>) data)
  "set data of matrix"
  (setf (ptr-cpu m) (cffi:foreign-alloc
		     :float :count (* (rows m) (cols m))
		     :initial-element 0.0))
  (attach-cpu-finalizer m)
  (setf (current-ptr m) 'cpu)
  (dotimes (r (rows m) m)
    (dotimes (c (cols m))
      (setf (cffi:mem-aref (ptr-cpu m) :float (+ (* (rows m) c) r))
	    (aref data (+ (* (cols m) r) c))))))

(defmethod gpu ((m <matrix>))
  "push c data to cuda data"
  (if (eq (current-ptr m) 'cpu)
      (progn (if (not (ptr-gpu m))
		 (progn (setf (ptr-gpu m)
			      (cffi:foreign-alloc :pointer))
			(cublasAlloc (* (rows m) (cols m))
				     (cffi:foreign-type-size ':float)
				     (ptr-gpu m))
			(attach-gpu-finalizer m)
			(setf (current-ptr m) 'gpu)))
	     (assert (eq :CUBLAS_STATUS_SUCCESS
			 (cublasSetMatrix (rows m) (cols m)
					  (cffi:foreign-type-size ':float)
					  (ptr-cpu m) (rows m)
					  (cffi:mem-ref (ptr-gpu m) ':pointer)
					  (rows m))))))
  m)

(defmethod cpu ((m <matrix>))
  "pull cuda data to c data"
  (if (eq (current-ptr m) 'gpu)
      (assert (eq :CUBLAS_STATUS_SUCCESS
		  (cublasGetMatrix (rows m) (cols m)
				   (cffi:foreign-type-size ':float)
				   (cffi:mem-ref (ptr-gpu m) ':pointer)
				   (rows m) (ptr-cpu m) (rows m)))))
  (setf (current-ptr m) 'cpu)
  m)

(defmethod print-object ((m <matrix>) stream)
  "prints a matrix, uh, as a list for now"
  (cpu m)
  (dotimes (r (rows m) m)
    (dotimes (c (cols m))
      (format stream "~2,1,6$ " (cffi:mem-aref (ptr-cpu m) :float
					  (+ (* (rows m) c) r))))
    (format stream "~%")))

(defmethod zeros (r c)
  (let ((Z (matrix r c)))
    (set-data Z (make-array (* r c) :initial-element 0.0))))

(defmethod ones (r c)
  (let ((Z (matrix r c)))
    (set-data Z (make-array (* r c) :initial-element 1.0))))

(defmethod eye (r)
  "Returns the identity matrix of size r x r"
  (let ((Z (zeros r r)))
    (dotimes (i r Z)
      (setf (cffi:mem-aref (ptr-cpu Z) :float (+ (* r i) i)) 1.0))))

(defmethod multiply-to ((A <matrix>) (B <matrix>) (Z <matrix>))
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

(defmethod multiply ((A <matrix>) (B <matrix>))
  "Multiply matrices A and B, preallocating returned Z"
  (assert (= (cols A) (rows B)))
  (let ((Z (zeros (rows A) (cols B))))
    (multiply-to A B Z)))

(defmethod add ((A <matrix>) (B <matrix>))
  "Computes the sum of two matrices of the same size"
  (assert (and (= (rows A) (rows B)) (= (cols A) (cols B))))
  (let ((Z (zeros (rows A) (cols A))))
    (dotimes (r (rows A) Z)
      (dotimes (c (cols A))
	(setf (cffi:mem-aref (ptr-cpu (cpu Z)) :float (+ (* (rows Z) c) r))
	      (+ (cffi:mem-aref (ptr-cpu (cpu A)) :float (+ (* (rows A) c) r))
		 (cffi:mem-aref (ptr-cpu (cpu B)) :float (+ (* (rows B) c) r))))))))

(defmethod rand (r c)
  "Generate uniform random matrix"
  (let ((Z (zeros r c)))
    (dotimes (i (* r c) Z)
      (setf (cffi:mem-aref (ptr-cpu (cpu Z)) :float i) (random 1.0)))))

(defmethod randn (r c)
  "Generate (CLT) normal random matrix using "
  (let ((Z (zeros r c)))
    (dotimes (i (* r c) Z)
      (setf (cffi:mem-aref (ptr-cpu (cpu Z)) :float i)
	    (- (reduce '+ (loop for ii from 1 to 12 collect (random 1.0))) 6)))))
