cd /media/christina/Data/ANFASS_data/partnet_data/dataset

myfiles=$(ls | grep train | grep txt | grep -v shuffle)

myarrayfiles=($myfiles)

for filename in "${myarrayfiles[@]}"; do
  linecount=$(cat $filename | wc -l)
  echo "${filename}   ${linecount}"
done
