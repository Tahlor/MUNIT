tar xvfz glibc-2.14.tar.gz
cd glibc-2.14
mkdir build
cd build
../configure --prefix=/fslhome/tarch/libraries/glibc-2.14
make all
make install
export LD_LIBRARY_PATH=/fslhome/tarch/libraries/glibc-2.14/lib:$LD_LIBRARY_PATH


/fslhome/tarch/glibc/glibc-2.14/build

export LD_LIBRARY_PATH=/fslhome/tarch/libraries/glibc-2.14/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/fslhome/tarch/libraries/glibc-2.14/lib


echo $LD_LIBRARY_PATH

munit
conda activate munit

python train.py --config configs/edges2handbags_folder.yaml

##  Get GLIBC version:
	/lib/libc.so.6	
	ldd --version

	
salloc --mem 32000M --time 1:00:00 --gres=gpu:1 --partition=m9


nvidia-smi
