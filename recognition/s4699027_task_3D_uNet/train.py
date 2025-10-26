#!/usr/bin/env python
"""
Script to train, validate, and test on data obtained from dataset.py for the 
3D Improved UNet. Model based on architecture found under modules.py. A 
variety of accuracy metrics (loss, accuracy, and multiclass dice and dice 
similarity coefficients) are used to evaluate the performance of the 
segmentation model, with visualisations of accuracy produced for clarity. The 
model and it's results are saved and logged for comparison.

@author Connor Davis
"""