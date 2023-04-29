# script to generate legal default parameter values for each model type (e.g.
# combinations of parameters that do not generate computational errors, to be
# used if the user does not specify parameters)

import numpy as _np
import pandas as _pd
import RegressionVisualiser3D 

# ==============================================================================
# GENERATE LEGAL DEFAULT PARAMATER COMBINATIONS

# set which model type to generate legal params for ("linear", "poisson" or
#  "binary_logistic")
type_string = 'binary_logistic'

# set how many default parameter combinations should be generated
desired_n_parameter_combos = 20

# an empty dataframe to store the legal parameter combinations
df = _pd.DataFrame({'intercept': [],
                    "predictor1_slope": [],
                    "predictor2_slope": [],
                    "error_sd": []})

# until the desired number of combinations are generated...
while len(df) < desired_n_parameter_combos:

    # assume the current combination is allowed (until proven otherwise)
    allowed = True

    # generate random parameters, appropriate for each model type
    if type_string == 'linear':
        params = _pd.DataFrame({"intercept" :  [_np.round(_np.random.uniform(-10, 10), 2)],
                "predictor1_slope" :   [_np.round(_np.random.uniform(-2, 2), 2)],
                "predictor2_slope" :  [_np.round(_np.random.uniform(-2, 2), 2)],
                "error_sd" : [_np.round(_np.random.uniform(1, 4), 2)]})
        
    elif type_string == 'poisson':
        params = _pd.DataFrame({"intercept" :  [_np.random.uniform(0, 0.1)],
            "predictor1_slope" :  [_np.random.uniform(-0.06, 0.06)],
            "predictor2_slope" : [_np.random.uniform(-0.06, 0.06)],
            "error_sd" : [_np.random.uniform(0.1, 0.5)]})
        
    elif type_string == 'binary_logistic':
        params = _pd.DataFrame({"intercept" :  [_np.random.uniform(0, 0.1)],
            "predictor1_slope" :  [_np.random.uniform(-0.07, 0.07)],
            "predictor2_slope" : [_np.random.uniform(-0.07, 0.07)],
            "error_sd" : [_np.random.uniform(0.1, 0.5)]})

    # check if the combination produces computational errors...
    try:
        RegressionVisualiser3D.model_plot(model_type = type_string+'_regression',
                    intercept = params['intercept'].values,
                   predictor1_slope = params['predictor1_slope'].values,
                   predictor2_slope = params['predictor2_slope'].values,
                   error_sd = params["error_sd"].values)
    # ... disallow it if it does
    except Exception:
        allowed = False

    # the parameter combination is legal, append it to the dataframe
    if allowed != False:
        df = _pd.concat([df, params])
        print("Found", len(df), "out of", desired_n_parameter_combos, "legal combos.")

# save the dataframe once desired number of combinations has been achieved
df.to_csv("legal_default_parameters/"+type_string+"_legal_default_params.csv", index = False)