git pull

nvcc ex2_2.cu --allow-unsupported-compiler --extended-lambda -o exo2_2
nvcc ex2_1.cu --allow-unsupported-compiler --extended-lambda -o exo2_1

./exo2_1
./exo2_2

rm exo2_1 exo2_2