git pull

if [ $# -eq 0 ]
then
    git add *
    git status
    git commit -m bridged
    git push
elif [ "$1" = "-c" ]
then
    name=""
    for var in  "$@"
    do
        if [ "$var" != "-c" ]
        then
            name="$name $var"
        fi
    done
    echo "compile $name" 
    nvcc $name --allow-unsupported-compiler --extended-lambda -o exo
    ./exo
    rm exo
elif [ "$1" = "-b" ]
then
    ssh -X -l alexandre.di-santo -p 2200$2 gpgpu.image.lrde.iaas.epita.fr
fi


# kinit
