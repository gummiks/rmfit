from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rmfit',
      version='0.1.0',
      description='Fit Rossiter McLaughlin Data',
      long_description=readme(),
      url='https://github.com/gummiks/rmfit/',
      author='Gudmundur Stefansson',
      author_email='gummiks@gmail.com',
      install_requires=['emcee','batman','radvel','pyde','corner'],
      packages=['rmfit'],
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      keywords='Astronomy',
      include_package_data=True
      )
