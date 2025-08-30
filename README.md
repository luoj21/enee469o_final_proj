## ENEE469O - Introduction to Optimization Final Project Code
Authors: _Jason Luo_, _Denis Fon_

Code for Convex Non-Negative Matrix Factorization based on [Ding-Li-Jordan Paper](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf) with applications to image compression and single channel audio source separation.

Given a non-negative matrix $$X$$, non-negative matrix factorization (NMF) attempts to factor $$X$$ into two non-negative matricies through minimizing the following objective:

$$\lVert X - WH \rVert _F^{2} \quad \text{Subject to} \quad  W\geq 0,H\geq 0$$

The objective of Convex NMF is formalized similarly:

$$\lVert X - XWG^T \rVert _F^{2} \quad \text{Subject to} \quad  W\geq 0, G^T\geq 0$$

Where $$\lVert . \rVert _F$$ denotes the Frobenius norm, $$F=XW$$ and the columns of $$F$$ are $$f_i = x_1w_1 + ... + x_nw_n$$, where $$w_i > 0$$

-----

Python version: ```Python 3.10.5```

- Do ```git clone https://github.com/luoj21/enee469o_final_proj.git```
- Create a virtual environment: ```python -m venv .venv```
- Then do ```source .venv/bin/activate``` (if on MacOS or Linux).
- Then ```pip install -r requirements.txt```
- The necessary notebooks used to simulate the convex/standard NMF comparisons are in the ```analysis``` folder

-----

References:
- [Audio Source Separation with NMF](https://medium.com/@zahrahafida.benslimane/audio-source-separation-using-non-negative-matrix-factorization-nmf-a8b204490c7d)
- [Audio Source Separation with NMF in PyTorch](https://gormatevosyan.com/audio-source-spearation-with-non-negative-matrix-factorization/)
- [MATLAB Implementation of the Ding-Li-Jordan Convex NMF](https://github.com/colinvaz/nmf-toolbox/blob/master/convexnmf.m). The ```convexnmf.m``` file in this repo is actually broken
- [MATLAB NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary/blob/master/solver/convex/convex_mu_nmf.m)
- [Image Compression With NMF](https://github.com/akcarsten/Non_Negative_Matrix_Factorization)
- [General paper on NMF](https://papers.nips.cc/paper\_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
- [Convex analysis of NMF](https://linjianma.github.io/pdf/NMF_227B_final_report.pdf)

-----