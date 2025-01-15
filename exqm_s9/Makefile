all: ex1 ex2

# -----------------------------------------------

ex1: ex1.cu ex1_main.cu
	nvcc --extended-lambda ex1.cu ex1_main.cu -o ex1

ex2: ex2.cu ex2_main.cu
	nvcc --extended-lambda ex2.cu ex2_main.cu -o ex2

# -----------------------------------------------

.PHONY: clean

clean:
	rm -f ex1 ex2
