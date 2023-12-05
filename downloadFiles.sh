#!/bin/bash
# download the files used for optimization:
#  1) B1+ data files
#  2) VOP files
url="https://keeper.mpdl.mpg.de/f/432e8e08fb284730b357/?dl=1"
fname="data.tar"
wget "$url" -O "$fname"
tar -xf ./data.tar

