#!/usr/bin/bash

orig_dir="/home/jev/hdd/memtacs/pilot/mri/"
cd $orig_dir
subjs=($(ls .))

for subj in "${subjs[@]}"
do
   if [ "$[subj]" = "MT-YG-120" ]
   then
     continue
   fi
   echo "$subj"
   cd "$orig_dir$subj"
   filename=$(ls *.nii)
   recon-all -subjid $subj -i $filename -all -parallel
done
cd "$orig_dir"
