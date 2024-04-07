# Multi-Class Anomaly Detection for Astronomical Transients

Paper Link: https://arxiv.org/abs/2403.14742

243rd AAS Meeting iPoster: https://aas243-aas.ipostersessions.com/default.aspx?s=16-F9-A7-80-27-2E-87-DF-6A-01-E4-07-E4-A2-07-2C&guestview=true

PyPI Package: https://pypi.org/project/astromcad/

Additional Figures from the paper are in the Figures/ folder (fully updated to reflect most recent model).

Code for training and plot generation in AstroMCAD.ipynb. Use this for plot generation and the python package for everything else.
If you have any questions, shoot me an email at rithwikca2020@gmail.com.


# `astromcad` Package Documentation

## Overview

For sample usage, see: https://colab.research.google.com/drive/1SrYVt9PAjai0NeLr4cI40Lriz_ruLhLt?usp=sharing

The `astromcad` package implements the transient anomaly detection methodology presented in https://arxiv.org/abs/2403.14742 and allows anyone to train custom light curve anomaly detectors. In short, the methodology is to repurpose the penultimate layer of a neural network classifier for anomaly detection. Then, to extract anomalies from this latent space, a separate isolation forest is trained on the observations from each class, and the minimum score from any detector is used as the final anomaly score. This isolation forest approach is called `MCIF` (Multi-Class Isolation Forest).

## Installation

<br>

Type `pip install astromcad` in your command line or terminal

<br>

## Usage

`astromcad` has 3 major classes, which are `Detect`, `Custom`, and `mcif`

### Detect

This class allows you to use the trained model from the research work for anomaly detection. The model from the work was trained from ZTF simulations which are described in more detail in the manuscript. 

Sample Usage

```python
from astromcad.astromcad import Detect

Detect.init() # Load the pretrained model

Detect.classify(light_curves, host_gals) # Uses the pretrained classifier to get a classification output
Detect.anomaly_score(light_curves, host_gals) # Generates the anomaly score for the given simple
``` 

The input `host_gals` must be an array of shape (N_SAMPLES, 2). Each 2-element entry should be [host redshift, milky way extinction] of that object. 

The input `light_curves` must be an array of shape (N_SAMPLES, 656, 4). For each sample, the 2-dimensional array should store the [median passband wavelength, scaled time since trigger, flux / 500, and flux error / 500] for each observation. If there are fewer than 656 observations, you can call

```python
x_data = Detect.pad(x_data)
```

You can also generate a real-time anomaly score plot for any transient. 

```python
Detect.plot_real_time(light_curve, host_gal)
```

Note that this function takes a single light curve, not a list of samples.

### Custom

This class allows you to create a custom light curve classifer to then use for anomaly detection. Start by creating your classifer

```python
from astromcad.astromcad import Custom

det = Custom(656, 4, 2, 9, 12) # n_timesteps, n_features per time step, n_host, latent_size, n_classes
det.create_model()
det.train(X_train, y_train, X_val, y_val, host_gal_train, host_gal_val) # EarlyStopping is initalized in the class
det.create_encoder()
det.init_mcif(X_train, y_train) # Values and labels to init MCIF with
det.score(light_curves, host_gals)
```

NOTE: If n_host is set to 0 during initalization, don't pass host galaxy information to any other function (or pass None). The class will not use the host galaxy information.

In the case that you want to create your own classifer, be sure to name the input layers and latent layer something memorable. To initalize a `Custom` object with this classifier, use 

```python
det.custom_model(model, lc_name, context_name, host_name) # Names of the respective layers
det.create_encoder()
...
```

To use the trained classifier, you can use `det.classify(light_curves, host_gals)`. To plot a real-time score evoluation, use

```python
det.plot_real_time(lc, unique_passbands, time, flux, flux_error, host_gal, names=['g', 'r'], colors=['g', 'r'])
```

`unique_passbands` is a list of all the unique values in the first column of lc. `time`, `flux`, and `flux_error` are the unscaled time, flux, and flux error values (likely to also be columns in lc). `names` and `colors` signify what plot label/color to be associated with each passband in `unique_passbands`.


### MCIF

This is an implementation of the Multi-Class Isolation Forest Algorithm, which trains a separate isolation forest for each class of data and uses the minimum score from any detector as the final anomaly score. MCIF is very simple to use.

```python
from astromcad.astromcad import mcif

multi = mcif(n_estimators=100)
mutli.train(x_data, labels)
multi.score(x_data)
multi.score_discrete(x_data)
```

`.score(x_data)` returns the minimum anomaly score, while `.score_discrete` reports the anomaly score per detector in a list. To see which unique entry in `labels` each element of the output refers to, use `multi.classes`

Example:

```python
multi.train(x_data, labels) # labels = [Class 1, Class 3, Class 2, Class 1, ...]
multi.score_discrete(x_data) # [0.45, -0.3, 0.2]
multi.labels # 				   [Class 1, Class 2, Class 3]
```




