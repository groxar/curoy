#! /bin/bash

#config
#change this to your username@address.com
host=

#script
directory=$(readlink -fn -- "$(pwd)")
name=$(basename $0)
target=~${directory#$HOME}
folder=$(dirname $target)


if [ -d $1 ] ; then
    ssh $host "mkdir -p $folder "
    rsync -zr $directory $host:$folder 
    ssh $host "cd $target && make -B" 
else 
    echo "$1 is not valid";
    exit 1
fi
