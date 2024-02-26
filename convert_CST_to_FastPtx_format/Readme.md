# Convert CST data into a dataset suitable for FastPtx

## Description
This script allows to convert the B1+ fields from a CST simulation to the (matlab-based) format required by the FastPtx toolbox

## Setup
Setup a python virtual environment and install the required packages from the requirements.txt file, for example like this:
```
python -m venv ./env
source ./env/bin/activate
pip install -r ./requirements.txt
```
If you want example data, you can download some here:
https://keeper.mpdl.mpg.de/f/ce3c8082b17f40c9be5f/?dl=1

## Running
- Put the data into inputData/CSTdata (see the example data linked above)
- open a jupyter lab or similar
- run `conversion.ipynb`

## Required data
- H-fields for each channel from CST, in the files "H_Ch1.h5" until "H_Ch8.h5"
- a SAR map (for creating a mask) in the file "SAR_CP_Mode.h5"
- A matlab file of the voxel model with the following data:
    - brainMask (3D volume with brain mask)
    - tissue Mask (3D volume with tissue mask)
    - deltaB0_Hz (simulated B0 field, set to 0 if you don't want this)

## Working principle
This code reads the B1+ fields from the CST data, as well as a SAR map. The SAR map is used to create a tissue mask.

The tissue mask is then used to coregister the voxel model information (B0, brainMask) to the CST data.

Afterwards everything is saved in the format required by FastPtx.

