from setuptools import setup, find_packages


setup(
    name="balu3",
    version="1.0",
    author='Domingo Mery',
    author_email='domingo.mery@uc.cl',
    url="https://github.com/domingomery/balu3",
    py_modules = ['balu3.fx.geo',
                  'balu3.io.plots',
                  'balu3.io.misc',
                  'balu3.io.visualization'
                  ]
)