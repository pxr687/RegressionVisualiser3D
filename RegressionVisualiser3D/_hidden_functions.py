import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import statsmodels.api as _sm
import sys as _sys
from IPython.display import display as _display, Markdown as _Markdown

# ==============================================================================
# CLASSES

# a custom error
class ComputationalError(Exception):
     pass

# ==============================================================================
# HIDDEN FUNCTIONS & GLOBAL VARIABLES

def _computation_error_string(e):
    "Hidden function for generating a string reporting computation errors."
    string = f"""
This combination of parameters (slopes etc.) generated computational errors!
Please try another combination... The error generated was:\n {e}
"""
    return string

def _get_legal_default_param_combos_for_current_model(intercept, predictor1_slope,
                                                       predictor2_slope, error_sd, 
                                name_of_legal_combos_csv_for_current_model_type):
    """
    Hidden function to get legal parameter combinations from csv. E.g. if the
    user has not supplied intercept, slope values, these will be loaded from
    a csv file containing legal values for the current model type.
    """

    # if some parameters any not supplied by the user
    if (intercept == None) | (predictor1_slope == None) | (predictor2_slope == None) | (error_sd == None): 

        # load in the dataframe containing legal default parameter combinarions
        legal_params = _pd.read_csv(f"legal_default_parameters/{name_of_legal_combos_csv_for_current_model_type}")

        # get a random row
        legal_params = legal_params.iloc[_np.random.choice(legal_params.index)]

    # set default intercept, slopes and error sd if none are provided by the user
    if intercept == None:
        intercept = legal_params["intercept"]

    if predictor1_slope == None:
        predictor1_slope = legal_params["predictor1_slope"]

    if predictor2_slope == None:
        predictor2_slope = legal_params["predictor2_slope"]

    if error_sd == None:
        error_sd = legal_params["error_sd"]

    return intercept, predictor1_slope, predictor2_slope, error_sd,

def _set_angles(axes, view_angle=None, view_elevation=None):
    """Hidden function to set the view angle and elevation on plots."""
    # set the view_angle and elevation, if the user has supplied them
    if view_angle is not None and view_elevation is None:
        axes.view_init(azim=view_angle)
    if view_elevation is not None and view_angle is None:
        axes.view_init(elev=view_elevation)
    if view_elevation is not None and view_angle is not None:
        axes.view_init(azim=view_angle, elev=view_elevation)

def _show_table(mod, model_type, show_statsmodels, verbose, intercept,
                predictor1_slope, predictor1_name, predictor2_slope,
                predictor2_name, interaction_slope, outcome_variable_name,
                markdown):
    """
    Hidden function to print out the regression table and true population
    regression equation.
    """
    # show the regression table?
    if show_statsmodels is True:
        print()
        _display(mod.summary())

        # print the true regression equation?

        if verbose == True:
            intercept = _np.round(intercept, 4)
            predictor1_slope =  _np.round(predictor1_slope, 4)          
            predictor2_slope = _np.round(predictor2_slope, 4)
            interaction_slope = _np.round(interaction_slope, 4)

        # ===========================================
        # LINEAR REGRESSION PRINTOUTS
        if (verbose == True) & (model_type == "linear_regression") & (markdown == True):
            print()
            _display(_Markdown("\nTrue population regression equation:"))
            print()
            _display(_Markdown(f"$Y = {intercept} + {predictor1_slope} * X_1 + {predictor2_slope} * X_2 + {interaction_slope} * X_1 * X_2 + error$"))
            print()
            _display(_Markdown("Where:"))
            _display(_Markdown(f" $Y$: {outcome_variable_name}"))
            _display(_Markdown(f" $X_1$: {predictor1_name}"))
            _display(_Markdown(f" $X_2$: {predictor2_name}\n"))
            
        if (verbose == True) & (model_type == "linear_regression") & (markdown == False):
            print()
            print("\nTrue population regression equation:")
            print()
            print(f"Y = {intercept} + {predictor1_slope} * X_1 + {predictor2_slope} * X_2 + {interaction_slope} * X_1 * X_2 + error")
            print()
            print("Where:")
            print(f" Y: {outcome_variable_name}")
            print(f" X_1: {predictor1_name}")
            print(f" X_2: {predictor2_name}\n")

        # ===========================================
        # POISSON REGRESSION PRINTOUTS
        if (verbose == True) & (model_type == "poisson_regression") & (markdown == True):
            print()
            _display(_Markdown("True population regression equation:"))
            print()
            _display(_Markdown(f"$ln(Y) = {intercept} + {predictor1_slope} * X_1 + {predictor2_slope} * X_2 + {interaction_slope} * X_1 * X_2 + error$"))
            print()
            _display(_Markdown("Where:"))
            _display(_Markdown(f" $Y$: {outcome_variable_name}"))
            _display(_Markdown(f" $X_1$: {predictor1_name}"))
            _display(_Markdown(f" $X_2$: {predictor2_name}"))

        if (verbose is True) & (model_type == "poisson_regression") & (markdown == False):
            print()
            print("True population regression equation:")
            print()
            print(f"ln(Y) = {intercept} + {predictor1_slope} * X_1 + {predictor2_slope} * X_2 + {interaction_slope} * X_1 * X_2 + error")
            print()
            print("Where:")
            print(f" Y: {outcome_variable_name}")
            print(f" X_1: {predictor1_name}")
            print(f" X_2: {predictor2_name}")

        # ===========================================
        # LOGISTIC REGRESSION PRINTOUTS

        if (verbose is True) & (model_type == "binary_logistic_regression") & (markdown == True):
            print()
            _display(_Markdown("True population regression equation:"))
            print()
            _display(_Markdown(f"$logit(Y) = {intercept} + {predictor1_slope} * X_1 + {predictor2_slope} * X_2 + {interaction_slope} * X_1 * X_2 + error$"))
            print()
            _display(_Markdown("Where:"))
            _display(_Markdown(f" $Y$: {outcome_variable_name}"))
            _display(_Markdown(f" $X_1$: {predictor1_name}"))
            _display(_Markdown(f" $X_2$: {predictor2_name}"))

        if (verbose is True) & (model_type == "binary_logistic_regression") & (markdown == False):
            print()
            print("True population regression equation:")
            print()
            print(f"logit(Y) = {intercept} + {predictor1_slope} * X_1 + {predictor2_slope} * X_2 + {interaction_slope} * X_1 * X_2 + error")
            print()
            print("Where:")
            print(f" Y: {outcome_variable_name}")
            print(f" X_1: {predictor1_name}")
            print(f" X_2: {predictor2_name}")

def _generate_population_x_y_datapoints(x_axis_min, x_axis_max, y_axis_min,
                                        y_axis_max,population_size,
                                        predictor_correlation_weight):
    "Hidden function to generate population data points."

    # generate the population datapoints
    x_noise = _np.random.randint(0, 10)
    pop_data_x = _np.random.choice(_np.linspace(
        x_axis_min*0.8, x_axis_max*0.8, 32), size=population_size) + _np.random.normal(0, x_noise, population_size)

    y_noise = _np.random.randint(0, 5)

        # predictors are orthogonal is predictor_correlation_weight == 0
    if predictor_correlation_weight == 0:
        pop_data_y = _np.random.choice(_np.linspace(y_axis_min*0.8, y_axis_max*0.8,
            32), size=population_size) + _np.random.normal(0, y_noise, population_size)
    else:
        pop_data_y = _np.random.choice(_np.linspace(y_axis_min*0.8, y_axis_max*0.8,
            32), size=population_size) + _np.random.normal(0, y_noise, population_size) + predictor_correlation_weight * pop_data_x + _np.random.normal(0, y_noise, population_size)

    return pop_data_x, pop_data_y

# some strings for printouts
def _explantation_string(population_size, sample_size):
    "Hidden function for generating a string explaining what model_plot() does."

    string = f"""
A population of {population_size} observations has been generated. A random 
sample of {sample_size} observations has been drawn from that population. 
Two graphs have been created.\n
The lefthand graph shows the population data and the population regression 
surface (e.g. if a regression model were fit to all of the population data).\n
The righthand graph shows the sample which was randomly drawn from the 
population. It also shows the sample regression surface (e.g. from a 
regression model fit to the sample data)."""

    return string

# string variable for a print out
_table_string = """
Beneath is the regression table (with slopes and p-values etc.) from the sample
data. The true population regression equation used to generate the data is also
shown."""