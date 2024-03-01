git pull
dir_path=""
if [ $# -eq 0 ]
then
    git add *
    git status
    git commit -m bridged
    git push
elif [ $# -eq 1 ]
then
    name=$dir_path$1
    echo "compile $name" 
    nvcc $name --allow-unsupported-compiler -o exo
    ./exo
else
    ssh -X -l alexandre.di-santo -p 2200$2 gpgpu.image.lrde.iaas.epita.fr
fi
