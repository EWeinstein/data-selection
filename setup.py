import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-selection",
    version="0.0.1",
    author="Eli Weinstein",
    author_email="eli.n.weinstein@gmail.com",
    description="Code associated with the paper: 'Bayesian Data Selection', E. Weinstein and J. Miller, JMLR, 2023",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EWeinstein/data-selection",
    packages=setuptools.find_packages(),
    install_requires=['autograd==1.3',
                      'pymanopt==0.2.5',
                      'numpy==1.21.1',
                      'scipy==1.7.1',
                      'matplotlib>=3.3.4',
                      'torch==1.9.0',
                      'pyro-ppl==1.7.0',
                      'pytest==6.2.5'],
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        'Topic :: Scientific/Engineering :: Bayesian Statistics',
    ],
    python_requires='>=3.9,<3.10',
    keywords=('bayesian-statistics probabilistic-programming ' +
              'machine-learning'),
)
