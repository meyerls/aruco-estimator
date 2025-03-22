"""
Monkey-patches the pycolmap package to add a 'model_name' property
that returns self.model.name, fixing the colmap-wrapper issue.
"""

import pycolmap._core as core


# Define a property for model_name
@property
def model_name(self):
    return self.model.name


# Patch the actual Camera class from pycolmap's C extension
core.Camera.model_name = model_name
