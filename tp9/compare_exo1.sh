git pull

nvcc ex1_2.cu --allow-unsupported-compiler --extended-lambda -o exo2
nvcc ex1_3.cu --allow-unsupported-compiler --extended-lambda -o exo3

./exo2
./exo3

rm exo2 exo3