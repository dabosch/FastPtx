# FastPtx: a python pTx pulse design tool for freely optimizing RF and gradient pulses with autodifferentiation

## Author:

Dario Bosch \<dario.bosch@tuebingen.mpg.de>

## Other Contributors:

- Qi Wang \<qi.wang@tuebingen.mpg.de>
	- helped with porting the bloch simulation code from C++ to Python
	- helped with pytorch support
- Alexander Loktyushin \<aloktyus@tuebingen.mpg.de>
	- gave valueable input on optimization with pytorch
    - helped working out bugs

# How to use

The following steps describe using Python Virtual Environments.
If you prefer using Anaconda you can try adapting those steps. I don't see why it shouldn't work

1. Open a terminal in the projects' main directory
1. Create a python virtual environment:
`python -m venv ./env`
1. Activate the environment:
`source ./env/bin/activate`
1. Install the necessary python packages
	- `pip install -r ./requirements.txt`
1. Get the example data and put it into the directory ./data/ by running the downloadFiles.sh script
    - `./downloadFiles.sh`
1. open either `spyder` or `jupyter lab` from the active shell
1. modify and run `calc_smallFA_paper.py`
    - the line `dev = torch.device('cuda')` sets the calculation to happen on a GPU. If you don't have a CUDA-enabled GPU, set it to `cpu` instead.
    - `do_UP = False` switches between tailored and universal pulses
    - Flip angle, pulse duration, etc can also be controlled by changing the settings in the beginning of the file.
    - Run by either executing the `calc_smallFA_paper.ipynb` notebook in jupyter or by running the `calc_smallFA_paper.py` script directly
1. You can create an animation (gif) for an optimized pulse using the `animatePulse.ipynb` notebook

# Citing
If you use this code, please cite the corresponding paper:
   
   Bosch, D.; Scheffler, K.: FastPtx: A Versatile Toolbox for Rapid, Joint Design of pTx RF and Gradient Pulses Using Pytorch's Autodifferentiation. Magnetic Resonance Materials in Physics, Biology and Medicine
