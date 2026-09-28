# Advanced Retail Sales Data Cleaning

This section contains an advanced data cleaning workflow using a retail store sales dataset. The goal was to inspect, clean, transform, and prepare messy transactional data for analysis.

## Dataset

The dataset contains retail transaction records, including:

- Transaction ID
- Customer ID
- Category
- Item
- Price Per Unit
- Quantity
- Total Spent
- Payment Method
- Location
- Transaction Date
- Discount Applied

## Cleaning Tasks

This workflow includes:

- Initial data inspection
- Missing value review by percentage
- Duplicate checks
- Categorical consistency checks
- Transaction date type correction
- Investigation of missing values by category, location, and payment method
- Missing discount handling
- Decision Tree imputation for missing item values
- KNN imputation for missing numeric values
- Recalculation of total spent
- IQR outlier detection
- MinMax normalization
- Skewness comparison before and after cleaning
- Boxplot review for numerical columns
- Export of cleaned dataset

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Notes

This project was completed as part of data analytics coursework and later reorganized as a portfolio section. Some modeling techniques were used for learning and practice purposes, with notes included to explain the cleaning decisions.
