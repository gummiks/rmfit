from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rmfit',
      version='1.0.2',
      description='Fit Rossiter McLaughlin Data',
      long_description=readme(),
      url='https://github.com/gummiks/rmfit/',
      author='Gudmundur Stefansson',
      author_email='gummiks@gmail.com',
      install_requires=['emcee','batman-package','radvel','corner','pandas','h5py','pytransit','arviz','celerite'],
      #packages=['rmfit'],
      packages=find_packages(),
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      keywords='Astronomy',
      include_package_data=True
      )
