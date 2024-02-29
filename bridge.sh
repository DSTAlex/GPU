echo $#
if [ $# -eq 0 ]
then
    git add *
    git status
    git commit -m bridged
    git push
else
    echo "pull"
fi