#! /bin/bash

#config
#change this to your username@address.com
host=teco 

#script
directory=$(readlink -fn -- "$1")
name=$(basename $1)
target=~${directory#$HOME}
folder=$(dirname $target)

if [ -d $1 ] ; then
    ssh teco "mkdir -p $folder "
    rsync -zr $1 $host:$folder 
    ssh $host "cd $target && make -B" 
else 
    echo "$1 is not valid";
    exit 1
fi
