function count_samples() {
  dataset=$1
  myfiles=$(ls | grep $dataset | grep txt | grep -v shuffle)

  myarrayfiles=($myfiles)

  for filename in "${myarrayfiles[@]}"; do
    linecount=$(cat $filename | wc -l)
    echo "${filename}   ${linecount}"
  done
}

function count_labels() {
  myfiles=$(ls | grep level-3)

  myarrayfiles=($myfiles)

  for filename in "${myarrayfiles[@]}"; do
    linecount=$(cat $filename | wc -l)
    echo "${filename}   ${linecount}"
  done
}

cd /media/christina/Data/ANFASS_data/partnet_data/dataset

count_samples train
echo
count_samples test
echo
count_samples val
echo

cd /media/christina/Data/ANFASS_data/partnet_data/partnet_dataset_master/stats/after_merging_label_ids

count_labels
