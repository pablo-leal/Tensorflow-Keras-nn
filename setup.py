from setuptools import setup, find_packages

# Parameters for Google Cloud Machine Learning Engine
setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Running Keras on google cloud ml-engine',
      author='Pablo Leal',
      author_email='pleal@nd.edu',
      license='MIT',
      install_requires=[
            'keras',
            'h5py',
      ],
      zip_safe=False)
