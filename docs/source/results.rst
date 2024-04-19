=================
Benchmark Results
=================

Here we give an overview of the benchmark results for the different models and datasets. The results are given in terms of the mean squared error (MSE) for each model and dataset. The MSE is computed as the average of the squared differences between the predicted and true values. The lower the MSE, the better the model performs.



----------------------
Single Step Prediction
----------------------

The table below shows the MSE for the different models and datasets for single step prediction for 900 points in the dataset. The MSE values are given in scientific notation.
The CNN, ResNet and NeuralPDE models have been trained and evaluated on the grid version of the datasets, while the other models have been trained and evaluated on the cloud version of the datasets.


.. csv-table:: Single Step Prediction
   :file: tables/results_singlestep.csv
   :header-rows: 1

----------------------------
Multi-16-Step  Prediction
----------------------------

The table below shows the MSE for the different models and datasets after 16 prediction steps (in a closed loop) for 900 points in the dataset. The MSE values are given in scientific notation.
The CNN, ResNet and NeuralPDE models have been trained and evaluated on the grid version of the datasets, while the other models have been trained and evaluated on the cloud version of the datasets.


.. csv-table:: 16-Step Prediction
   :file: tables/results_multistep.csv
   :header-rows: 1