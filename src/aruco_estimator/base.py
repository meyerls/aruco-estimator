#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""


# Built-in/Generic Imports
# ...

# Libs
# ...

# Own modules
# ...


class ScaleFactorBase(object):
    def __init__(self):
        """
        Base class for scale factor estimation.

            ---------------
            |    Detect   |
            ---------------
                    |
                    v
            ---------------
            |     Run     |
            ---------------
                    |
                    v
            ---------------
            |   Evaluate  |
            ---------------
                    |
                    v
            ---------------
            |     Apply   |
            ---------------
        """
        pass

    def __evaluate(self):
        return NotImplemented

    def __detect(self):
        return NotImplemented

    def run(self):
        return NotImplemented

    def apply(self, *args, **kwargs):
        return NotImplemented
