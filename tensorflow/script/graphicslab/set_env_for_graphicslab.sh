scripts_path=/root/miniconda3/envs/OCNN/etc/conda
mkdir -p "$scripts_path/activate.d"
cp activate.sh "$scripts_path/activate.d/"
chmod 777 "$scripts_path/activate.d/activate.sh"
mkdir -p "$scripts_path/deactivate.d"
cp deactivate.sh "$scripts_path/deactivate.d/"
chmod 777 "$scripts_path/deactivate.d/deactivate.sh"


#NOTE:
#on this PC there is one miniconda under /root and one miniconda under /home/graphicslab

#NOTE:
#This was used to manage to build ocnn with cuda 10.1 and tensorflow 1.14
#https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0
#also this was used to make sure it is build using gcc and g++ version 8
#https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa
