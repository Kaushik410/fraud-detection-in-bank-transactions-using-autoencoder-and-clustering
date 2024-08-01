
# Bank Transaction Anomaly Detection and Clustering

This project uses an autoencoder for feature extraction, Isolation Forest for anomaly detection, and K-Means clustering to identify patterns in bank transaction data.

## Prerequisites

- Python 3.6 or higher
- pandas
- numpy
- scikit-learn
- keras
- seaborn
- matplotlib

## Installation

You can install the required Python packages using pip:

```bash
pip install pandas numpy scikit-learn keras seaborn matplotlib
```

## Usage

1. **Prepare the Data:**

   Load the data from the Excel file and fill missing values for withdrawal and deposit amounts with 0.

   ```python
   file_path = '/path/to/your/bank.xlsx'
   data = pd.read_excel(file_path)
   data.loc[:, 'WITHDRAWAL AMT'] = data['WITHDRAWAL AMT'].fillna(0)
   data.loc[:, 'DEPOSIT AMT'] = data['DEPOSIT AMT'].fillna(0)
   ```

2. **Normalize the Data:**

   Define the features for the autoencoder and normalize the data.

   ```python
   features = ['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']
   X = data[features]
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Split the Data:**

   Split the data into training and test sets, retaining indices.

   ```python
   X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=42)
   X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
   ```

4. **Train the Autoencoder:**

   Define the autoencoder architecture and train it.

   ```python
   input_dim = X_train.shape[1]
   encoding_dim = 2
   input_layer = Input(shape=(input_dim,))
   encoded = Dense(encoding_dim, activation='relu')(input_layer)
   decoded = Dense(input_dim, activation='sigmoid')(encoded)
   autoencoder = Model(inputs=input_layer, outputs=decoded)
   autoencoder.compile(optimizer='adam', loss='mean_squared_error')
   autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
   ```

5. **Feature Extraction and Anomaly Detection:**

   Extract features using the encoder part of the autoencoder and use Isolation Forest for anomaly detection.

   ```python
   encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=1).output)
   X_train_encoded = encoder.predict(X_train_scaled)
   X_test_encoded = encoder.predict(X_test_scaled)
   iso_forest = IsolationForest(contamination=0.01, random_state=42)
   iso_forest.fit(X_train_encoded)
   y_pred_train = iso_forest.predict(X_train_encoded)
   y_pred_test = iso_forest.predict(X_test_encoded)
   anomalies_train = np.where(y_pred_train == -1)
   anomalies_test = np.where(y_pred_test == -1)
   ```

6. **Clustering:**

   Perform K-Means clustering on the encoded features.

   ```python
   kmeans = KMeans(n_clusters=3, random_state=42)
   kmeans.fit(X_train_encoded)
   clusters_train = kmeans.predict(X_train_encoded)
   clusters_test = kmeans.predict(X_test_encoded)
   ```

7. **Visualization:**

   Visualize the clusters and anomalies.

   ```python
   sns.scatterplot(data=data, x='Encoded Feature 1', y='Encoded Feature 2', hue='Cluster', style='Anomaly', palette='deep')
   plt.title('Clusters and Anomalies in Encoded Feature Space')
   plt.show()
   ```

## Output

The results will be saved in a new Excel file `bank_with_anomalies_and_clusters.xlsx`.

```python
data.to_excel('bank_with_anomalies_and_clusters.xlsx', index=False)
```

## Visualization

Additional visualizations are provided to analyze the anomalies and clusters over time and by feature distributions.

```python
# Create time series plot of anomalies
plt.figure(figsize=(14, 7))
anomalies_per_day = data.groupby(data['DATE'].dt.date)['Anomaly'].sum()
plt.plot(anomalies_per_day.index, anomalies_per_day.values, marker='o', linestyle='-')
plt.title('Anomaly Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Anomalies')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create heatmap of anomalies by date and cluster
data['Date'] = data['DATE'].dt.date
heatmap_data = data.pivot_table(index='Date', columns='Cluster', values='Anomaly', aggfunc='sum', fill_value=0)
plt.figure(figsize=(14, 7))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Heatmap of Anomalies by Date and Cluster')
plt.xlabel('Cluster')
plt.ylabel('Date')
plt.tight_layout()
plt.show()
```

## License

This project is licensed under the MIT License.
