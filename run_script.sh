#! /bin/bash

videos_folder="pallet_videos"
full_path="$( pwd)/$videos_folder"
echo "$full_path"
echo "Files in pallet_videos folder:"
for file in $( ls $videos_folder); do
    echo "$file"
    filename=(${file//./ })
	name=${filename[0]} 
	
	python find_signature.py -v $name
done
echo "-----"
