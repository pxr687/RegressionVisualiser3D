# RegressionVisualiser3D
RegressionVisualiser3D is a simple single-function plotting toolbox for visualizing linear regression, Poisson regression and binary logistic regression in 3D, for teaching purposes.

A notebook demonstrating the package is here: https://github.com/pxr687/RegressionVisualiser3D/blob/master/RegressionVisualiser3D_Demo.ipynb 

The user supplies parameters for the data-generating process and a population of observations is created through those parameters. The population data consists of two continuous predictor variables and one outcome variable. The type of outcome variable depends on the type of regression model being visualized (continuous for linear regression, binary for logistic regression etc.)..

A random sample (of a size specified by the user) is drawn from the population data. A regression model is then fit to the sample data.

3D visualisations are then shown, which depict:

* The population data
* The population regression surface
* The sample data
* The sample regression surface

An optional regression table (with slopes and p-values etc.) is also shown, alongside the true regression equation used to generate the data. 

To aid understanding, the user can also specify the names of the predictor variables and the outcome variable.

If the user does not supply population parameters, defaults are used.
