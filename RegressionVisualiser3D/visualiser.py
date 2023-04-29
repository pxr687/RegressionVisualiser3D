import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import statsmodels.api as _sm
import sys as _sys
from IPython.display import display as _display, Markdown as _Markdown
from ._hidden_functions import (ComputationalError, _computation_error_string,
                              _set_angles, _show_table,
                              _generate_population_x_y_datapoints,
                             _explantation_string, _table_string)
from ._linear_regression import _linear_regression
from ._poisson_regression import _poisson_regression
from ._binary_logistic_regression import _binary_logistic_regression

# ==============================================================================
# USER-FACING FUNCTIONS

def model_plot(model_type="linear_regression",
               intercept=None,
               predictor1_slope=None,
               predictor2_slope=None,
               interaction_slope=0,
               predictor_correlation_weight = None,
               sample_size=100,
               population_size=1000,
               error_sd=None,
               predictor1_name="Predictor 1",
               predictor2_name="Predictor 2",
               outcome_variable_name="Outcome Variable",
               x_axis_min =  None,
               x_axis_max =  None,   
               y_axis_min = None,
               y_axis_max =  None,   
               show_statsmodels=True,
               legend_loc="lower center",
               view_angle=None,
               view_elevation=None,
               plot_size=(16, 8),
               data_alpha=1,
               surface_alpha=0.7,
               verbose = True, 
               markdown = True):
    """
    This function generates a 3D visualization of several types of 
    regression model. It creates population data, based on parameters 
    supplied as arguments. It then draws a random sample from that 
    population data. A regression model is then fit to the sample data.
    The type of regression model which is fit is specified by the 
    `model_type" argument. N.B. the `model_type` parameter also specifies
    the form of the data-generating process (e.g. `linear_regression` will
    generate population data from a linear equation, the other models involve
    an equation with a link function).

    The regression surfaces and the data are then shown on 3D plots, for
    the population and the sample data. By default, explanations of 
    what the function is doing/showing are printed out for the user.

    The user can set all of the parameters, BUT some combinations may 
    cause computational errors, depending on the type of regression 
    model.

    Parameters
    ----------

    model_type: a string specifying the type of regression model 
    to be visualized. Must be one of "linear_regression", 
    "poisson_regression" or "binary_logistic_regression"

    intercept: the intercept to be used in the population data
    generating process. Default is None, in which case a value will be
    randomly selected.

    predictor1_slope: the slope of the first predictor, to be used in 
    the population data generating process. Default is None, but if 
    Default is None, in which case a value will be randomly selected.

    predictor2_slope: the slope of the second predictor, to be used in 
    the population data generating process. Default is None, in which case a 
    value will be randomly selected.

    interaction_slope: the slope of the interaction between the two 
    predictors to be used in the population data generating process. 
    Default is None, in which case a value will be randomly selected.

    predictor_correlation_weight: sets how correlated the predictors will be.
    Default is None, which results in a weight randomly selected between -0.4 
    and 0.4

    sample_size: an integer setting the size of the random sample, 
    drawn from the population data. Default is 100.

    population_size: an integer setting the size of the population. 
    Default is 1000.

    error_sd: the standard deviation of the error added to the 
    population regression equation used in the data generating process. 
    Default is None, but if None is supplied a default will be 
    chosen based on model_type.

    predictor1_name: a string setting the name of the first predictor. 
    Default is "Predictor 1".

    predictor2_name: a string setting the name of the second predictor. 
    Default is "Predictor 2".

    outcome_variable_name:  a string setting the name of the outcome 
    variable. Default is "Outcome Variable".

    x_axis_min: a number setting the smallest possible value of the x axis 
    datapoints. Default is a random number.

    x_axis_max: a number setting the largest possible value of the x axis 
    datapoints. Default is a random number.

    y_axis_min: a number setting the smallest possible value of the y axis 
    datapoints. Default is a random number.

    y_axis_max: a number setting the largest possible value of the y axis 
    datapoints. Default is a random number.

    show_statsmodels: a Boolean, if True, the regression table from 
    the regression model fit to the sample data will be shown.

    legend_loc: a string setting the location to display the figure 
    legend. Default is "lower center".

    view_angle: a number setting the view angle for the graphs. Default 
    is None, which will use the matplotlib default.

    view_elevation: a number setting the view elevation for the graphs. 
    Default is None, which will use the matplotlib default.

    plot_size: a tuple setting the figure size. Default is (16,8).

    data_alpha: a number between 0 and 1, setting the see-through-ness 
    of the datapoints on the graphs. Default is 1 (opaque).

    surface_alpha: a number between 0 and 1, setting the 
    see-through-ness of the regression surfaces on the graphs. 
    Default is  0.7

    verbose: a Boolean. If True an explanation of what the function 
    is doing will be displayed alongside the graphs (via printouts).
    The true regression equation used in the data generating process
    will also be shown. Default is True.

    markdown: a Boolean. Set to True if running in a Jupyter notebook, and text
    will be printed in nice-looking markdown. If False, the print() function 
    will be used (and won't look as good!). Default is True.

    Returns
    -------
    None.
    """

    # randomize the correlation between the predictors, if none is specified by
    # the user
    if predictor_correlation_weight is None:
        predictor_correlation_weight = _np.round(_np.random.uniform(-0.3, 0.3), 2)

    # an array of the legal model types
    model_types = _np.array(
        ["linear_regression", "poisson_regression", "binary_logistic_regression"])

    # if verbose mode is true
    if verbose == True:

        # remove the "_" from the model type name
        formatted_model_type = model_type.replace("_", " ")

        # print out an explanation of what the function is doing/showing
        if markdown == True:
            _display(_Markdown(f"\n3D {formatted_model_type} visualiser: "))
            _display(_Markdown(_explantation_string(population_size, sample_size)))
        elif markdown == False:
            print(f"\n3D {formatted_model_type} visualiser: ")
            print(_explantation_string(population_size, sample_size))

        # if the statsmodels table should be shown, let the user know where it 
        # will be displayed
        if (show_statsmodels == True) & ( markdown == True):
            _display(_Markdown(_table_string))
        elif (show_statsmodels == True) & ( markdown == False):
            print(_table_string)

    # check the model_type supplied is legal, raise error if not
    assert model_type in model_types, "The model_type you have specified is" + \
        "not recognized! It should be one of:" + str(model_types)
    
# set the axis limits, if none are supplied

    if x_axis_min == None:
        x_axis_min = _np.random.randint(20, 100)
    
    if x_axis_max == None:
        x_axis_max = x_axis_min + abs(_np.random.randint(1, 100))

    if y_axis_min == None:
        y_axis_min = _np.random.randint(20, 100)
    
    if y_axis_max == None:
        y_axis_max = y_axis_min + abs(_np.random.randint(1, 100))

# ==============================================================================
# LINEAR REGRESSION

    if model_type == model_types[0]:
        _linear_regression(intercept,
                           predictor1_slope,
                           predictor2_slope,
                           interaction_slope,
                           predictor_correlation_weight,
                           sample_size,
                           population_size,
                           error_sd,
                           predictor1_name,
                           predictor2_name,
                           outcome_variable_name,
                           x_axis_min ,
                           x_axis_max ,   
                           y_axis_min ,
                           y_axis_max ,   
                           show_statsmodels,
                           legend_loc,
                           view_angle,
                           view_elevation,
                           plot_size,
                           data_alpha,
                           surface_alpha,
                           verbose,
                           markdown)

# ==============================================================================
# POISSON REGRESSION

    if model_type == model_types[1]:
         _poisson_regression(intercept,
                           predictor1_slope,
                           predictor2_slope,
                           interaction_slope,
                           predictor_correlation_weight,
                           sample_size,
                           population_size,
                           error_sd,
                           predictor1_name,
                           predictor2_name,
                           outcome_variable_name,
                           x_axis_min ,
                           x_axis_max ,   
                           y_axis_min ,
                           y_axis_max ,   
                           show_statsmodels,
                           legend_loc,
                           view_angle,
                           view_elevation,
                           plot_size,
                           data_alpha,
                           surface_alpha,
                           verbose,
                           markdown)

# ==============================================================================
# BINARY LOGISTIC REGRESSION

    if model_type == model_types[2]:
        _binary_logistic_regression(intercept,
                           predictor1_slope,
                           predictor2_slope,
                           interaction_slope,
                           predictor_correlation_weight,
                           sample_size,
                           population_size,
                           error_sd,
                           predictor1_name,
                           predictor2_name,
                           outcome_variable_name,
                           x_axis_min ,
                           x_axis_max ,   
                           y_axis_min ,
                           y_axis_max ,   
                           show_statsmodels,
                           legend_loc,
                           view_angle,
                           view_elevation,
                           plot_size,
                           data_alpha,
                           surface_alpha,
                           verbose,
                           markdown)