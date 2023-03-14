from setuptools import find_packages, setup
setup(
	name='SAXSsimulations',
	packages=find_packages(),      
    version='0.0.1',
	description='Simulate SAXS experiment',
	long_description='file: README.md',
	url='https://github.com/sofyalaski/SAXS-simulations',
	author='sofyalaskina',
	author_email='sofyalaskina@gmail.com',
	license='MIT',
	classifiers=[
		'Natural Language :: English',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
	],
	install_requires=['torch', 'numpy', 'sasmodels', 'scipy', 'freia'],
	python_requires='>3.6'
)
