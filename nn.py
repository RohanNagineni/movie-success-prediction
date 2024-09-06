import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import RandomOverSampler

# Load the data
file_path = 'training_data.csv'  # Update this with your file path
data = pd.read_csv(file_path)

# Drop the specified columns
columns_to_drop = ['name', 'vote_average', 'vote_count', 'revenue', 'tmdb_id', 'imdb_id']
data_cleaned = data.drop(columns=columns_to_drop)

# Replace -999 and -1 with NaN for normalization
data_cleaned = data_cleaned.apply(lambda x: x.replace([-999, -1], np.nan) if x.dtype == 'float' else x)

# Normalize the data excluding the 'success' column
features = data_cleaned.drop(columns=['success'])
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features.fillna(0))

# Prepare the final dataset
X = pd.DataFrame(normalized_features, columns=features.columns)
y = data_cleaned['success'].values

# Balance the data by upsampling the minority class
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# Print the number of training examples after oversampling
print(f'Number of training examples after oversampling: {len(y_res)}')
print(f'Number of class 0 examples after oversampling: {(y_res == 0).sum()}')
print(f'Number of class 1 examples after oversampling: {(y_res == 1).sum()}')

# Split the balanced data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define the model creation function with dropout and regularization
def create_model(learning_rate=0.001, optimizer='adam', dropout_rate=0.5, regularization=0.01):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(regularization)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(regularization)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(regularization)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the KerasClassifier
model = KerasClassifier(
    model=create_model,
    verbose=1
)

# Define the grid search parameters
param_grid = {
    'model__learning_rate': [0.01, 0.001],
    'model__dropout_rate': [0.2, 0.5],
    'model__regularization': [0.01, 0.001],
    'fit__epochs': [50],
    'fit__batch_size': [32, 64]
}

# Define KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# Display the best parameters and the best score
print(f'Best parameters: {grid_result.best_params_}')
print(f'Best score: {grid_result.best_score_}')

# Create the final model with the best parameters
best_params = grid_result.best_params_
final_model = create_model(
    learning_rate=best_params['model__learning_rate'],
    dropout_rate=best_params['model__dropout_rate'],
    regularization=best_params['model__regularization']
)

# Train the final model on the full training data with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = final_model.fit(X_train, y_train, validation_split=0.2, epochs=best_params['fit__epochs'], batch_size=best_params['fit__batch_size'], callbacks=[early_stopping], verbose=1)

# Evaluate the final model on the test set
loss, accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the final model
final_model.save('final_model.keras')

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Load the model (for testing purpose)
# loaded_model = load_model('final_model.keras')
# loaded_scaler = joblib.load('scaler.pkl')
