import setuptools 

setuptools.setup(
      name="dynabench",
      version='0.2.0',
      description='Benchmark dataset for learning dynamical systems from data',
      author='Andrzej Dulny',
      author_email='andrzej.dulny@protonmail.com',
      licence='',
      packages=['dynabench'],
      package_dir={'': '.'},
      package_data={'dynabench': ['config/*.yaml']},
      install_requires=[
        "requests", "tqdm", "numpy>=1.23.5", "h5py>=3.9.0"
      ],
      extras_require={
        "dynabench": ["dedalus>=3"],
    })
