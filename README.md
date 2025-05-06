Code for Convex Non-Negative Matrix Factorization based on [Ding-Li-Jordan Paper](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf) and applications to single channel audio source separation.

Python version: ```Python 3.10.5```

Do ```git clone https://github.com/luoj21/enee469o_final_proj.git```

Create a virtual environment: ```python -m venv .venv```

Then do ```source .venv/bin/activate``` (if on MacOS or Linux).

Then ```pip install -r requirements.txt```

Other references:
- [Audio Source Separation with NMF](https://medium.com/@zahrahafida.benslimane/audio-source-separation-using-non-negative-matrix-factorization-nmf-a8b204490c7d)
- [Audio Source Separation with NMF in PyTorch](https://gormatevosyan.com/audio-source-spearation-with-non-negative-matrix-factorization/)
- [MATLAB Implementation of the Ding-Li-Jordan Convex NMF](https://github.com/colinvaz/nmf-toolbox/blob/master/convexnmf.m). The ```convexnmf.m``` file in this repo is actually broken I think