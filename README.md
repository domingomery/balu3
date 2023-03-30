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

`fx.geo` -- Geometric features

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


`fx.chr` -- Chromatic features

| Function    | Description                    |
| ----------- | ------------------------------ |
| basicint    | Basic intensity features       |
| hugeo       | Hu moments                     |
| lbp         | Local binary patterns          |
| hog         | Histogram of gradients         |
| haralick    | Haralick texture features      |
| gabor       | Gabor features (by Balu)       |
| sci_gabor   | Gabor features (by scimage)    |
| fourier     | Fourier features (DFT)         |
| dct         | Cosine transform features (DCT)|
| clt         | Crossing line profile          |
| contrast    | Contrast features              |


## `ft` -- Feature Transformation

`ft.norm` -- Normalization

| Function    | Description                    |
| ----------- | ------------------------------ |
| minmax      | MinMax normalization           |
| mean0       | mean=0, std=1 normalization    |


`ft.trans` -- Linear transformation

| Function    | Description                    |
| ----------- | ------------------------------ |
| pca         | Principal component analysis   |



## `fs` -- Feature Selection

`fs.sel` -- Normalization

| Function    | Description                    |
| ----------- | ------------------------------ |
| jfisher     | Fisher score                   |
| sp100       | specifity=1 score              |
| clean       | Cleaning                       |
| sfs         | Sequential forward selection   |
| exsearch    | Exhaustive search selection    |


## `io` -- Input/Output

`io.misc` -- Miscellaneous

| Function    | Description                    |
| ----------- | ------------------------------ |
| dirfiles    | Files of a drectory            |
| num2fixstr  | Number to string with fixed 0s |
| imageload   | Load of an image with indices  |


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




