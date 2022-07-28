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
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires=[
          'matplotlib',
          'numpy',
          'sentence_transformers',
          'tensorflow.keras',
          'tensorflow_probability'
      ],
      include_package_data=True,
      zip_safe=False)
