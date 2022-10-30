from setuptools import setup, find_packages

setup(name='meta',
      version='0.2.0.1',
      description='Metaembedding tool',
      url='http://github.com/anasampa/metaembedding',
      author='Ana',
      author_email='email',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          'tqdm >= 4.29.1',
          'scikit-learn>=0.18',
          'scikit-image>=0.12',
          'pyDOE2==1.3.0',
          'sentence_transformers',
          #'lime @ https://github.com/anasampa/lime/archive/vector_emb.zip',
          'lime @ https://github.com/anasampa/lime/archive/master.zip',  
          'ktrain'
      ],
      extras_require={
          'dev': ['pytest', 'flake8'],
      },
      include_package_data=True,
      zip_safe=False)
