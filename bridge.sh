git pull
dir_path=""
if [ $# -eq 0 ]
then
    git add *
    git status
    git commit -m bridged
    git push
else
    name=$dir_path$1
    echo "compile $name" 
    nvcc $name --allow-unsupported-compiler -o exo
    ./exo
fi