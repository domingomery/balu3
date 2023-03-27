from setuptools import setup, find_packages


setup(
    name="balu3",
    version="1.0",
    authors='Domingo Mery, Christian Pieringer, Marco Bucchi',
    author_email='domingo.mery@uc.cl',
    url="https://github.com/domingomery/balu3",
    py_modules = ['balu3.fx.geo',
                  'balu3.fx.chr',
                  'balu3.ft.norm',
                  'balu3.fs.sel',
                  'balu3.im.proc',
                  'balu3.im.kfunc',
                  'balu3.im.seg',
                  'balu3.io.plots',
                  'balu3.io.misc',
                  'balu3.io.vis'
                  ]
)