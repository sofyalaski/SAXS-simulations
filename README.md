# Create 3D memory efficient Fourier Transformations with Python  to analyze scattering data.

1. SAXSsimulations folder contains:
    * files to simulate 3D density of sphere/ hard sphere / cylinder in the unit box 
    * read-in some density containing file 
    * function to fast fourier transform a 3D density structure with different methods
    * rebinning
    * comparing simulation to SasView simulation
    * example of usage is in the main directory -> `creating_forms.ipynb`
    * example how to read in a density file -> `real_density.ipynb`
2. ML_model folder contains machine learning models tested in the thesis (requires FrEIA):
    * `utils.py` file contains the class wrapper to read data, normalize it, train etc.
    * `losses.py` contains definitions of the loss functions
    * `monitoring.py` contains files to output loss decay information
    * `visualizations.py` contains methods to visualize the result of prediction
    * Notebooks with respective architectures and their training results 
3. tests contains scripts and csv files to reproduce the results section of the thesis with all the necessary plotting functions
4. `create_data.py` creates training set with SasView 
5. envoronment file
