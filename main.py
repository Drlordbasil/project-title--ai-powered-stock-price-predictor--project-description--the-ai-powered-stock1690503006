import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt


def collect_stock_data(stock_symbol):
    base_url = 'https://finance.yahoo.com/quote/'
    url = base_url + stock_symbol + '/history?p=' + stock_symbol
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Collect historical stock prices
    rows = soup.find_all('tr', {'class': 'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'})
    dates = []
    prices = []
    
    for row in rows:
        columns = row.find_all('td')
        date = pd.to_datetime(columns[0].text.strip())
        price = float(columns[4].text.strip().replace(',', ''))
        dates.append(date)
        prices.append(price)
    
    # Collect financial statements, news sentiment, etc.
    # ...
    
    return pd.DataFrame({'Date': dates, 'Price': prices})


def preprocess_data(data):
    # Remove missing values
    data = data.dropna()
    
    # Normalize numeric features
    scaler = MinMaxScaler()
    numeric_cols = ['Price']  # Add other numeric feature names
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Encode categorical variables
    # ...
    
    return data


def select_features(data):
    features = data.drop(['Price'], axis=1)
    targets = data['Price']
    
    # Perform feature selection using LASSO regression or other methods
    # ...
    
    return features, targets


def train_model(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    # Train random forest regressor
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)
    
    # Evaluate model performance
    scores = cross_val_score(rf_reg, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())
    
    return rf_reg, rmse


def train_lstm(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    # Reshape features for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Build LSTM model
    model = keras.Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train LSTM model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return model, rmse


def predict_stock_price(model, data):
    # Prepare data for prediction
    # ...
    
    # Make predictions using the trained model
    # ...
    
    return predictions


def visualize_results(data, predictions):
    plt.plot(data['Date'], data['Price'], label='Actual')
    plt.plot(data['Date'], predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()


# Main function
def main():
    stock_symbol = input("Enter stock symbol: ")
    
    # Step 1: Collect data
    data = collect_stock_data(stock_symbol)
    
    # Step 2: Data preprocessing
    preprocessed_data = preprocess_data(data)
    
    # Step 3: Feature selection
    features, targets = select_features(preprocessed_data)
    
    # Step 4: Train models
    rf_reg, rf_rmse = train_model(features, targets)
    lstm_model, lstm_rmse = train_lstm(features, targets)
    
    print("Random Forest RMSE:", rf_rmse)
    print("LSTM RMSE:", lstm_rmse)
    
    # Step 5: Perform stock price prediction
    # ...
    
    # Step 6: Visualization
    # ...
    

if __name__ == '__main__':
    main()