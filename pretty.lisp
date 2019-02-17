(in-package :cl-cublas)

(defparameter *CUBLAS_HANDLE* (cffi:foreign-alloc :pointer))
(assert (eq :CUBLAS_STATUS_SUCCESS
	    (cublasCreate_v2 *CUBLAS_HANDLE*)))

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
    :initform 'cpu)
   (op
    ;; Specifies whether the matrix should be traposed prior to an
    ;; operation, such as addition or multiplication
    :accessor op
    :initform :CUBLAS_OP_N)))

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
  "prints a matrix"
  (cpu m)
  (if (eq (op m) :CUBLAS_OP_N)
      (dotimes (r (rows m) m)
	(dotimes (c (cols m))
	  (format stream "~2,1,6$ " (cffi:mem-aref (ptr-cpu m) :float
						   (+ (* (rows m) c) r))))
	(format stream "~%"))
      (dotimes (r (rows m) m) ;; cols and rows are switched now
	(dotimes (c (cols m))
	  (format stream "~2,1,6$ " (cffi:mem-aref (ptr-cpu m) :float
						   (+ (* (cols m) r) c))))
	(format stream "~%"))))

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

(defmethod transpose ((A <matrix>))
  (if (eq (op A) :CUBLAS_OP_N)
      (setf (op A) :CUBLAS_OP_T)
      (setf (op A) :CUBLAS_OP_N))
  (let ((tmp (rows A)))
    (setf (rows A) (cols A))
    (setf (cols A) tmp))
  A)

(defmethod multiply-to ((A <matrix>) (B <matrix>) (Z <matrix>))
  "Multiply matrices A and B, storing result in Z (returned)"
  ;; Ensure that the inner dimensions of A and B match.
  (assert (= (cols A) (rows B)))
  ;; Ensure that the outer dimensions of A and B match the size of Z.
  (assert (and (= (rows A) (rows Z)) (= (cols B) (cols Z))))
  (let ((m (rows A)) (n (cols B)) (k (cols A)))
    (cffi:with-foreign-objects ((alpha ':float) (beta ':float))
      (setf (cffi:mem-ref alpha :float) 1.0)
      (setf (cffi:mem-ref beta :float) 0.0)
      (assert (eq :CUBLAS_STATUS_SUCCESS
      		  (cublasSgemm_v2 (cffi:mem-ref *CUBLAS_HANDLE* ':pointer)
      				  (op A) (op B) m n k alpha
      				  (cffi:mem-ref (ptr-gpu (gpu A)) ':pointer) m
      				  (cffi:mem-ref (ptr-gpu (gpu B)) ':pointer) k beta
      				  (cffi:mem-ref (ptr-gpu (gpu Z)) ':pointer) m)))))
  Z)

(defmethod multiply ((A <matrix>) (B <matrix>))
  "Multiply matrices A and B, preallocating returned Z"
  (assert (= (cols A) (rows B)))
  (let ((Z (zeros (rows A) (cols B))))
    (multiply-to A B Z)))


(defmethod add-to ((A <matrix>) (B <matrix>) (Z <matrix>))
  "Adds matrices A and B, storing result in Z (returned). This
   probably isn't much faster than doing it on the CPU, but prevents
   having to copy memory back and forth"
  ;; Ensure that the dimensions of A and B match"
  (let ((m (rows A)) (n (cols B)) (k (cols A)))
    (cffi:with-foreign-objects ((alpha ':float) (beta ':float))
      (setf (cffi:mem-ref alpha :float) 1.0)
      (setf (cffi:mem-ref beta :float) 1.0)
      (assert (eq :CUBLAS_STATUS_SUCCESS
      		  (cublasSgeam (cffi:mem-ref *CUBLAS_HANDLE* ':pointer)
      			       (op A) (op B) m n alpha
      			       (cffi:mem-ref (ptr-gpu (gpu A)) ':pointer) m beta
      			       (cffi:mem-ref (ptr-gpu (gpu B)) ':pointer) k
      			       (cffi:mem-ref (ptr-gpu (gpu Z)) ':pointer) m)))))
  Z)

(defmethod add ((A <matrix>) (B <matrix>))
  "Computes the sum of two matrices of the same size"
  (assert (and (= (rows A) (rows B))
	       (= (cols A) (cols B))))
  (let ((Z (zeros (rows A) (cols A))))
    (add-to A B Z))
  Z)

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
