# Code for "Enhanced Conformal Prediction for Deep Learning Based Time Series Forecasting" submitted to ICML2025
## abstract
Time series forecasting is critical in various applications, with deep learning-based point prediction methods achieving notable success. But there is a gap in how to obtain confidence interval about these methods. Existing methods for modeling confidence intervals, such as quantile regression or probabilistic forecasting, often suffer from performance issues and require costly model retraining. In this work, we propose a lightweight conformal prediction method based on existing point prediction models, enabling improved coverage and interval length without the need for retraining. Our method leverages the features extracted by deep learning models to predict confidence intervals, coupled with an adaptive mechanism to ensure validity under distribution shifts. Our theoretical analysis shows that the coverage of the proposed method converges to the desired level, with coverage error ties to feature quality of point prediction models.  Experimental results across 12 datasets demonstrate that our approach provides shorter confidence intervals while maintaining sufficient coverage rates. 

## how to run the method
FIrst, run predict.py to train point prediction model.
Seconr run main.py to obtian predicted confidence intervals
