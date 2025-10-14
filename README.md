# CATGNN
Code used for my publication:</br>
Al-Fahdi, M.; Rurali, R.; Hu, J.; Wolverton, C.; Hu, M. Accelerating Discovery of Extreme Lattice Thermal Conductivity by Crystal Attention Graph Neural Network (CATGNN) Using Chemical Bonding Intuitive Descriptors. arXiv preprint arXiv:2410.16066.
- please cite the above work if you use the code

## Required Packages
the following packages are required to run the code:</br>
<code>torch=2.5.1</code></br>
<code>torch-geometric=2.6.1</code></br>
<code>torch-scatter=2.1.2</code></br>
<code>e3nn=0.5.1</code></br>
<code>Jarvis-tools=2024.10.30</code></br>
<code>scikit-learn=1.2.2</code></br>

other versions might work, but those versions were successful in running the code

## Usage
1- untar the data directory by running:</br>
<code>tar -xvzf data.tar</code></br>
2- you can edit the model parameters from the file "model_params.yaml" and the parameters should be straightforward to edit.</br>
3- "id_prop.csv" is the file needed for training. 'id_prop_cohp.csv' and 'id_prop_cobi.csv' are for the normalized -ICOHP and normalized ICOBI, respectively.</br>
4- you can simply run the following line to run the code:</br>
<code>python main.py</code>

## Credit
* Please consider reading my published work in Google Scholar using this [link](https://scholar.google.com/citations?user=5tkWy4AAAAAJ&hl=en&oi=ao) thank you :)
* also please let me know if more features are needed to be added and/or improved 
