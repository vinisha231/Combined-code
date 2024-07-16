#!/bin/bash

main_directory="/data/CT_images/python_main_file"
main="$main_directory/main.py"
main_backup="$main_directory/main.py.bak"

cd $main_directory

cp $main $main_backup

for file in "${@:1}";
do
	file_insert_header="#>>>>>>>>>>$file>>>>>>>>>>"
	file_insert_footer="#<<<<<<<<<<$file<<<<<<<<<<"
	csplit "$main" "/$file_insert_header/+1" "/$file_insert_footer/"
	if [ $? -ne 0 ]; then
		echo "$file_insert_header" >> $main
		cat $file >> $main
		echo "$file_insert_footer" >> $main
	else
		mv xx00 $main
		cat $file >> $main
		cat xx02 >> $main
		rm -rf xx*
	fi
done
