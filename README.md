# Sales Forecasting Using VARMAX & LSTM

## Overview
This project aims to forecast sales for the Top 1 product families with the highest sales in a grocery store. The forecasting is based on historical data and observed trends. The dataset used contains sales data from a grocery store in Ecuador, South America. It includes sales records for around 33 different product categories, with a total of up to 3 million transactions.
  
For forecasting, I used two approaches: **VARMAX**, which helps identify relationships between multiple factors like sales, promotions, and external influences, and **LSTM**, a deep learning model that excels at recognizing complex time-series patterns.


## Dataset
- The dataset includes several important features that influence sales trends and forecasting:
  - **Date:** The timestamp for each sales record
  - **Sales:** Gives the total sales for a product family at a particular store at a given date.
  - **Onpromotion:** Gives the total number of items in a product family that were being promoted at a store at a given date.
  - **dcoilwtico:** Represents the daily values of the West Texas Intermediate(WTI) crude oil price index, which is important for tracking and analyzing trends in the oil market. Ecuador is an oil-dependent country and its economic health is highly vulnerable to shocks in oil prices. 

- Some features in the dataset have missing values, which can affect the model performance. I addressed this by polynomial interpolation and backward-fill to maintain continuity in trends.


## Tools & Technologies Used
- **Environment:** Google Colab  
- **Modeling Tools:**
  - **VARMAX** using statsmodels.tsa.statespace.varmax.VARMAX  
  - **LSTM** using tensorflow.keras


## Data Preprocessing
- Feature normalization/standardization.  
- Time transformations (lag features, differencing for VARMAX).  
- Splitting data into train and test sets.  


## Model Development
- **VARMAX:** Selecting optimal parameters (p, q), performance evaluation using MSE or AIC/BIC.  
- **LSTM:** Model architecture (number of layers, dropout, activation functions), hyperparameter selection (optimizer, batch size, epochs), callback functions such as EarlyStopping


## Model Evaluation & Comparison
- Evaluation metrics using RMSE, MAE, MAPE.  
- Performance comparison of VARMAX vs. LSTM.  
- Visualization of predictions vs. actual values.  
