import setuptools 

setuptools.setup(
      version='0.1.0',
      description='Benchmark dataset for learning dynamical systems from data',
      name="dynabench",
      packages=setuptools.find_packages('.'),
      package_dir={'': '.'},
      install_requires=[
        "numpy>=1.23.5", "py-pde>=0.27.1", "torch>=2.0.1", "scipy>=1.2.0", "torchdata>=0.6.0", 
        "torch_geometric>=2.3", "torch_cluster>=1.6"
      ]
     )