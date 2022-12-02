from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np


def run_setup():
    extensions = [
        Extension(
            name='pltreeshap._pltree',
            sources=['pltreeshap/_pltree.pyx'],
            include_dirs=[np.get_include()],
            define_macros=[('NPY_NO_DEPRECATED_API', 0)],
            language = 'c++',
        ),
    ]

    # Load Version
    with open("VERSION", "r") as f:
        version = f.read().rstrip()

    setup(
        name='pltreeshap',
        author='schufa-innovationlab',
        version=version,
        description='Computation of interventional SHAP (interaction) values for piecewise linear trees.',
        license='Apache License 2.0',
        packages=['pltreeshap'],
        ext_modules=cythonize(extensions, compiler_directives={'language_level': '3'}),
        install_requires=['numpy>=1.14.5'],
        python_requires='>=3.7',
        classifiers=[
            "Intended Audience :: Science/Research",
            'Intended Audience :: Developers',
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
    )


if __name__ == '__main__':
    run_setup()
