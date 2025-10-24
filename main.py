import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
data = pd.read_csv('data/house_data.csv')

# Step 2: Features (X) and Target (y)
X = data[['area', 'bedrooms', 'age']]
y = data['price']
print(X,y)
# Step 3: Split dataset → 60% train, 40% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(X_train,y_train)
# Step 4: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Show results
print("Predicted prices for test data:\n", predictions)
print("Actual prices:\n", y_test.tolist())

# Step 7: Check accuracy
score = model.score(X_test, y_test)
print(f"\nModel Accuracy: {score*100:.2f}%")
# Predict a new house price
new_data = pd.DataFrame([[4000, 5, 8]], columns=['area', 'bedrooms', 'age'])
predicted_price = model.predict(new_data)

print(f"Predicted price for new house: {predicted_price[0]:,.2f}")

# Save model after training
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
