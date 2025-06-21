import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your CSV file
df = pd.read_csv(r"TASK - 1\house_data.csv")  # Replace with your filename

# Drop rows with missing values in selected columns
df = df.dropna(subset=[
    'living_area', 'num_rooms', 'number_of_buildings_in_hectare',
    'travel_time_private_transport', 'population_in_hectare', 'price'
])

# Define input features (X) and target variable (y)
X = df[['living_area', 'num_rooms', 'number_of_buildings_in_hectare',
        'travel_time_private_transport', 'population_in_hectare']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Compare predicted vs actual prices
results = pd.DataFrame({
    "Living Area": X_test['living_area'],
    "Rooms": X_test['num_rooms'],
    "Actual Price": y_test,
    "Predicted Price": y_pred
})
print("\nPrediction Results:\n", results.head())
