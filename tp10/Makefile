all: ex1 ex2 ex3

# -----------------------------------------------

ex1: ex1.cu ex1_main.cu
	nvcc --extended-lambda ex1.cu ex1_main.cu -o ex1

ex2: ex2.cu ex2_main.cu
	nvcc --extended-lambda ex2.cu ex2_main.cu -o ex2

ex3: ex3.cu ex3_main.cu
	nvcc --extended-lambda ex3.cu ex3_main.cu -o ex3

# -----------------------------------------------

.PHONY: clean

clean:
	rm -f ex1 ex2 ex3
