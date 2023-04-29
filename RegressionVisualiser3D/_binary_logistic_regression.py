import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import statsmodels.api as _sm
import sys as _sys
from IPython.display import display as _display
from ._hidden_functions import (ComputationalError, _computation_error_string,
                              _set_angles, _show_table,
                              _generate_population_x_y_datapoints,
                             _explantation_string, _table_string, 
                             _get_legal_default_param_combos_for_current_model)

# a string for the model type
model_type = "binary_logistic_regression"

def _binary_logistic_regression(intercept=None,
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
               verbose=True,
               markdown = True):
    
    # see model_plot() in visualiser.py for documentation of input arguments

    # if some parameters any not supplied by the user, get default values that
    # will not cause computational errors
    intercept, predictor1_slope, predictor2_slope, error_sd,  = _get_legal_default_param_combos_for_current_model(
                                                                intercept, 
                                                                predictor1_slope,
                                                                predictor2_slope,
                                                                error_sd, 
                                                                "binary_logistic_legal_default_params.csv")

    # =============================================
    # GENERATING THE POPULATION DATA

    # generate the population data points
    pop_data_x, pop_data_y = _generate_population_x_y_datapoints(x_axis_min,
                                                                x_axis_max,
                                                                y_axis_min,
                                                                y_axis_max,
                                                                population_size,
                                                                predictor_correlation_weight)
    # create the linear prediction equation
    lin_pred = intercept + predictor1_slope*pop_data_x + predictor2_slope*pop_data_y + \
        interaction_slope*pop_data_x*pop_data_y + \
        _np.random.normal(0, error_sd, size=population_size)
    
    # use the linear predictor to get the population z values, based on the
    # slopes/intercept
    pop_data_z = (_np.exp(lin_pred))/(1 + _np.exp(lin_pred))
    pop_data_z = _np.where(pop_data_z >= 0.5, 1, 0)

    # get the sample data and fit a binary logistic regression with statsmodels
    population_df = _pd.DataFrame({predictor1_name: pop_data_x,
                                    predictor2_name: pop_data_y,
                                    outcome_variable_name: pop_data_z,
                                    predictor1_name+" * "+predictor2_name: pop_data_x * pop_data_y})
    
    # for the wireframe population regression surface
    x_axis_interval = _np.arange(_np.min(population_df[predictor1_name]), _np.max(population_df[predictor1_name]))
    y_axis_interval = _np.arange(_np.min(population_df[predictor2_name]), _np.max(population_df[predictor2_name]))

    # Create a meshgrid from the x and y intervals
    x, y = _np.meshgrid(x_axis_interval, y_axis_interval)
    lin_pop_z = intercept + predictor1_slope*x + \
        predictor2_slope*y + interaction_slope*x*y
    z = _np.exp(lin_pop_z)/(1 + _np.exp(lin_pop_z))

    # plot the population regression surface and data
    fig = _plt.figure(figsize=plot_size)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_wireframe(x, y, z, color="darkred",
                        label="population logistic regression surface",
                        alpha=surface_alpha)
    ax1.scatter(pop_data_x[pop_data_z >= 0.5], pop_data_y[pop_data_z >= 0.5],
                pop_data_z[pop_data_z >= 0.5], color="red", 
                label=outcome_variable_name+" = 1 (population)",
                alpha=data_alpha)
    ax1.scatter(pop_data_x[pop_data_z < 0.5], pop_data_y[pop_data_z < 0.5],
                pop_data_z[pop_data_z < 0.5],
                marker="x", color="red",
                label=outcome_variable_name+" = 0 (population)",
                alpha=data_alpha)
    ax1.set_zticks([0, 1])
    ax1.set_zlabel(outcome_variable_name+"/ Probability")
    ax1.set_zticks([0, 1])
    ax1.set_title("Population (N = "+str(population_size)+") :")
    _plt.xlabel(predictor1_name)
    _plt.ylabel(predictor2_name)
    _set_angles(ax1, view_angle=view_angle, view_elevation=view_elevation)

    # =============================================
    # DRAWING THE SAMPLE DATA

    sample_df = population_df.sample(n=sample_size)

    # try to fit the regression model, warn the user if they generate 
    # computation errors
    try:
        mod = _sm.Logit(sample_df[outcome_variable_name], _sm.add_constant(sample_df[[
                        predictor1_name, predictor2_name, predictor1_name+" * "+predictor2_name]])).fit()
    except Exception as e: 
        raise ComputationalError(_computation_error_string(e))

    # fit the regression model to the sample data
    mod = _sm.Logit(sample_df[outcome_variable_name], _sm.add_constant(sample_df[[
                    predictor1_name, predictor2_name, predictor1_name+" * "+predictor2_name]])).fit()

    # get the parameters from the regression model fit to the sample data
    mod_intercept = mod.params["const"]
    mod_predictor1_slope = mod.params[predictor1_name]
    mod_predictor2_slope = mod.params[predictor2_name]
    mod_interaction_slope = mod.params[predictor1_name +
                                        " * "+predictor2_name]

    # wireframe z values from the parameters from the regression model fit to the sample data
    sample_z = _np.exp(mod_intercept + mod_predictor1_slope*x + mod_predictor2_slope*y + mod_interaction_slope * x*y)/(
        1 + _np.exp(mod_intercept + mod_predictor1_slope*x + mod_predictor2_slope*y + mod_interaction_slope * x*y))

    # plot the sample regression surface and data
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_wireframe(x, y, sample_z, color="darkblue",
                        label="sample logistic regression surface", alpha=surface_alpha)
    ax2.scatter(sample_df[predictor1_name][sample_df[outcome_variable_name] >= 0.5],
                sample_df[predictor2_name][sample_df[outcome_variable_name] >= 0.5],
                sample_df[outcome_variable_name][sample_df[outcome_variable_name] >= 0.5],
                color="blue", label=outcome_variable_name+" = 1 (sample)",
                alpha=data_alpha)
    ax2.scatter(sample_df[predictor1_name][sample_df[outcome_variable_name] < 0.5],
                sample_df[predictor2_name][sample_df[outcome_variable_name] < 0.5],
                sample_df[outcome_variable_name][sample_df[outcome_variable_name] < 0.5],
                color="blue", marker="x", label=outcome_variable_name+" = 0 (sample)",
                alpha=data_alpha)
    ax2.set_zlabel(outcome_variable_name+"/ Probability")
    ax2.set_zticks([0, 1])
    ax2.set_title("Sample (n = "+str(sample_size)+") :")
    _plt.xlabel(predictor1_name)
    _plt.ylabel(predictor2_name)

    # set the view angle and elevation, if the user has supplied them
    _set_angles(ax2, view_angle=view_angle, view_elevation=view_elevation)

    fig.legend(loc=legend_loc)

    # show the regression table and true regression equation, if set to be
    # displayed
    _show_table(mod, model_type, show_statsmodels, verbose, intercept,
                predictor1_slope, predictor1_name, predictor2_slope,
                predictor2_name, interaction_slope, outcome_variable_name,
                markdown)

    # show the plot
    _plt.show()