
# Intermediate Machine Learning Course: Housing Price Prediction

This Kaggle course focuses on intermediate-level machine learning skills and applies them to predict housing prices using the dataset from the Housing Prices competition.

## Machine Learning Techniques Employed

In this competition, I employed a variety of machine learning techniques to optimize the predictive models. Here is an overview of my approach:

I defined five distinct random forest models, each with different hyperparameters, to identify the best-performing model for the competition:
```bash
from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
```

Subsequently, I created a function to evaluate and compare these models based on their mean absolute error:

```bash
from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```
This approach enabled me to assess the models' performance and identify the model with the lowest mean absolute error, thus improving the accuracy of housing price predictions.


