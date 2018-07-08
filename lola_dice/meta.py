"""Utility functions for meta-learning.

Some code was adapted from https://github.com/deepmind/learning-to-learn.

Note that `make_with_custom_variables` requires using `tf.get_variable` for all
variable creation purposes. Otherwise, `custom_getter` is not injected. Works
best with `sonnet` which guarantees that all internal variables are created
with `tf.get_variable`. For the same reason, does NOT work with `tf.layers`
consistently.
"""

import collections

import mock

import numpy as np
import tensorflow as tf


def wrap_variable_creation(func, custom_getter):
    """Provides a custom getter for all variable creations."""
    original_get_variable = tf.get_variable
    def custom_get_variable(*args, **kwargs):
        return original_get_variable(
            *args, custom_getter=custom_getter, **kwargs)

    # Mock the get_variable method
    with mock.patch("tensorflow.get_variable", custom_get_variable):
            return func()


def make_with_custom_variables(func, variables):
    """Calls func and replaces any trainable variables.

    This returns the output of func, but whenever `get_variable` is called it
    will replace any trainable variables with the tensors in `variables`, in the
    same order. Non-trainable variables will re-use any variables already
    created.

    Args:
        func: Function to be called.
        variables: A list of tensors replacing the trainable variables.

    Returns:
        The return value of func is returned.
    """
    variables = collections.deque(variables)
    custom_variables = {}

    def custom_getter(getter, name, **kwargs):
        if kwargs["trainable"]:
            if name not in custom_variables:
                custom_variables[name] = variables.popleft()
            return custom_variables[name]
        else:
            kwargs["reuse"] = True
            return getter(name, **kwargs)

    return wrap_variable_creation(func, custom_getter)
