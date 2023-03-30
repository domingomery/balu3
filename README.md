# WARNING: 

Work in Progress (do not install it !!)

# balu3

Python implementation for Balu Toolbox, based on previous implementations: Balu for Matlab (implemented by Domingo Mery), pyBalu for Python (implemented by Marco Bucchi), and pyXvis (implemented by Domingo Mery and Christian Pieringer).

# Authors

Domingo Mery, Christian Pieringer and Marco Bucchi

# Examples

[![Colab Examples](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/domingomery/patrones/blob/master/Notebooks.md) Examples implemented in Balu3 using Google Colab.


# Modules

## `fx` -- Feature Extraction

`fx.geo`

| Function    | Description                    |
| ----------- | ------------------------------ |
| basicgeo    | Basic geometric features       |
| hugeo       | Hu moments                     |
| flusser     | Flusser moments                |
| gupta       | Gupta moments                  |
| fourierdes  | Fourier descriptors            |
| efourierdes | Elliptic Fourier descriptors   |
| fit_ellipse | Elliptic feaures               |
| lbp         | Local Binary Patterns          |


* fs -- Feature Selection

* ft -- Feature Transformation

* io -- Input/Output 

* im -- Image Processing




# Requirements

- Python 3.6 or higher
- Numpy
- Scipy
- Matplotlib
- OpenCV 4.0 or higher

# Instalation
In the first installation use directly in the directory folder:

`pip install .`

Then, to upgrade:

`pip install --upgrade .`

Or:

`git clone https://github.com/domingomery/balu3`

`pip install ./balu3`




