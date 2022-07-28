from setuptools import setup, find_packages

VERSION = 0.0.1
DESCRIPTION = "Metaembedding tool."
LONG_DESCRIPTION = "A tool to combine and visualize embedding combinations for similarity prediction."
setup(name='metaembedding',
      version=VERSION,
      description=DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      url='http://github.com/anasampa/metaembedding',
      author='Ana',
      author_email='ana.sampa.oi@gmail.com',
      license='--',
      #packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      packages = find_packages()
      install_requires=[
          'matplotlib',
          'numpy',
          'sentence_transformers',
          'tensorflow.keras',
          'tensorflow_probability'
          #'scipy',
          #'tqdm >= 4.29.1',
          #'scikit-learn>=0.18',
          #'scikit-image>=0.12',
          #'pyDOE2==1.3.0'
      ],
      #extras_require={
        #  'dev': ['pytest', 'flake8'],
      #},
      include_package_data=True,
      zip_safe=False)