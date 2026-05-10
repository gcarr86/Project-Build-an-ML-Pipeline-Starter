# Model Card — NYC Airbnb Price Prediction Model

## Model Details
This model is a supervised regression model built using a RandomForestRegressor within a scikit-learn pipeline. The full pipeline includes preprocessing steps for missing values and categorical encoding.

The model was trained and tracked using MLflow and Weights & Biases as part of an end-to-end machine learning pipeline.

## Intended Use
This model predicts nightly Airbnb listing prices based on listing attributes.

It is intended for:
- Educational ML pipeline demonstration
- Exploratory pricing analysis

It should not be used for real financial or investment decisions.

## Training Data
NYC Airbnb listings dataset.

Features:
- minimum_nights
- number_of_reviews
- reviews_per_month
- calculated_host_listings_count
- availability_365
- neighbourhood_group
- room_type

Target:
- price

## Model Architecture
- Preprocessing:
  - Median imputation for numeric features
  - Most frequent imputation for categorical features
  - One-hot encoding for categorical variables
- Model:
  - RandomForestRegressor
  - Hyperparameters loaded from rf_config.json
  - Fixed random seed for reproducibility

## Evaluation Metrics
- Validation MAE: ~36.4
- Test MAE: ~37.4

Metric used: Mean Absolute Error (MAE)

## Ethical Considerations
- Model may reflect bias in Airbnb pricing across neighborhoods
- Should not be used for discriminatory pricing decisions

## Limitations
- Only trained on NYC data
- No text features used
- Sensitive to market changes and data drift
- May not generalize to other cities

## Reproducibility
- MLflow used for model tracking
- W&B used for artifact logging
- Deterministic random seed used

## Intended Output
Predicts nightly price of Airbnb listings in USD