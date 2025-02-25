file="../output/bgl"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

file="../output/bgl/bert"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi