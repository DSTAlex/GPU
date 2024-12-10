all: ex1_1 ex1_2 ex1_3   ex2_1 ex2_2

# -----------------------------------------------

ex1_1: ex1_1.cu
	nvcc ex1_1.cu -o ex1_1

ex1_2: ex1_2.cu
	nvcc ex1_2.cu -o ex1_2

ex1_3: ex1_3.cu
	nvcc ex1_3.cu -o ex1_3

# -----------------------------------------------

ex2_1: ex2_1.cu
	nvcc ex2_1.cu -o ex2_1

ex2_2: ex2_2.cu
	nvcc ex2_2.cu -o ex2_2

# -----------------------------------------------

.PHONY: clean

clean:
	rm -f ex1_1 ex1_2 ex1_3   ex2_1 ex2_2
