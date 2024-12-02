NVCC		:= nvcc
NVCC_FLAGS  := 

all: ex1 ex2

# -----------------------------------------------

ex1: ex1.cu broadcast.cu
	$(NVCC) $(NVCC_FLAGS) ex1.cu broadcast.cu -o ex1

ex2: ex2.cu reduce.cu
	$(NVCC) $(NVCC_FLAGS) ex2.cu reduce.cu -o ex2

# -----------------------------------------------

.PHONY: clean

clean:
	rm -f ex1 ex2
