cd /media/christina/Data/ANFASS_data/partnet_data/dataset

function count {
  dataset=$1
  myfiles=$(ls | grep $dataset | grep txt | grep -v shuffle)

  myarrayfiles=($myfiles)

  for filename in "${myarrayfiles[@]}"; do
    linecount=$(cat $filename | wc -l)
    echo "${filename}   ${linecount}"
  done
}

count train
echo
count test
echo
count val