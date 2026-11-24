# ==============================================================================
# Step 0: Import Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("--- Project: Federated Learning (REFIT & UK-DALE) ---")

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_weights(model): return model.get_weights()
def set_weights(model, weights): model.set_weights(weights)

def fed_avg_aggregate(global_model, client_models_list):
    global_weights = get_weights(global_model)
    new_weights = []
    for i in range(len(global_weights)):
        layer_weights = [get_weights(model)[i] for model in client_models_list]
        new_weights.append(np.mean(layer_weights, axis=0))
    set_weights(global_model, new_weights)
    return global_model

def build_base_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation='tanh', return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ==============================================================================
# Function to Run FL on a Dataset
# ==============================================================================
def run_federated_learning(data_path, house_files, dataset_name):
    print(f"\n{'='*70}")
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # 1. Load Data
    clients_data = {}
    for i, file_name in enumerate(house_files):
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            time_col = 'Time' if 'Time' in df.columns else ('datetime' if 'datetime' in df.columns else None)
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(by=time_col).reset_index(drop=True)
            clients_data[f'Client_{i+1}'] = df
            print(f"Loaded {file_name} | Rows: {df.shape[0]}")
    
    # 2. Preprocess
    def preprocess(df, dataset_name):
        df_processed = df.copy()
        if len(df_processed) > 400000:
            df_processed = df_processed.sample(n=400000, random_state=42).sort_index().reset_index(drop=True)
        if 'Issues' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Issues'])
        
        time_col = 'Time' if 'Time' in df_processed.columns else ('datetime' if 'datetime' in df_processed.columns else None)
        
        if dataset_name == 'REFIT':
            target_col = 'Aggregate' if 'Aggregate' in df_processed.columns else 'Appliance1'
        else:
            target_col = 'power'
        
        if time_col:
            df_processed['Hour'] = df_processed[time_col].dt.hour
            df_processed['DayOfWeek'] = df_processed[time_col].dt.dayofweek
            df_processed['Is_Weekend'] = df_processed['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            df_processed['Minute'] = df_processed[time_col].dt.minute
            df_processed['Lag_1'] = df_processed[target_col].shift(1)
            df_processed['Lag_5'] = df_processed[target_col].shift(5)
            df_processed['Lag_15'] = df_processed[target_col].shift(15)
            df_processed['Rolling_Mean_10'] = df_processed[target_col].rolling(window=10, min_periods=1).mean()
            df_processed['Rolling_Mean_60'] = df_processed[target_col].rolling(window=60, min_periods=1).mean()
            df_processed['Rolling_Std_10'] = df_processed[target_col].rolling(window=10, min_periods=1).std().fillna(0)
            df_processed = df_processed.dropna().reset_index(drop=True)
            df_processed = df_processed.drop(columns=[time_col])
        
        features_to_scale = df_processed.drop(columns=[target_col])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_to_scale)
        df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
        df_scaled[target_col] = df_processed[target_col].values
        return df_scaled, target_col

    client_processed_data = {}
    target_columns = {}
    for client_name in clients_data:
        processed_df, target_col = preprocess(clients_data[client_name], dataset_name)
        client_processed_data[client_name] = processed_df
        target_columns[client_name] = target_col
        print(f"Processed {client_name} | Target: {target_col}")
    
    # 3. Split
    client_train_data = {}
    client_test_data = {}
    for client_name, df in client_processed_data.items():
        target_col = target_columns[client_name]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        client_train_data[client_name] = (X_train, y_train)
        client_test_data[client_name] = (X_test, y_test)
        print(f"{client_name} ready: Train {X_train.shape} | Test {X_test.shape}")
    
    # 4. Build Model
    first_client = list(client_train_data.keys())[0]
    input_dim = client_train_data[first_client][0].shape[1]
    print(f"\nInput Dimension: {input_dim}")
    
    # 5. FL Simulation
    NUM_ROUNDS = 8
    EPOCHS = 5
    BATCH_SIZE = 512
    SAMPLE_SIZE = 60000
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    
    print(f"\n--- Starting Federated Learning ({dataset_name}) ---")
    global_model = build_base_model(input_dim)
    
    for round_num in range(1, NUM_ROUNDS + 1):
        local_models = []
        for client_name in client_train_data.keys():
            client_model = build_base_model(input_dim)
            set_weights(client_model, get_weights(global_model))
            X_train, y_train = client_train_data[client_name]
            if len(X_train) > SAMPLE_SIZE:
                idx = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
                X_s, y_s = X_train.iloc[idx], y_train.iloc[idx]
            else:
                X_s, y_s = X_train, y_train
            client_model.fit(X_s, y_s, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])
            local_models.append(client_model)
        global_model = fed_avg_aggregate(global_model, local_models)
        print(f"  Round {round_num}/{NUM_ROUNDS} completed.")
    
    # 6. Personalization & Evaluation
    print(f"\n--- Personalization & Evaluation ({dataset_name}) ---")
    personalized_models = {}
    for client_name in client_train_data.keys():
        pers_model = build_base_model(input_dim)
        set_weights(pers_model, get_weights(global_model))
        X_train, y_train = client_train_data[client_name]
        pers_model.fit(X_train[:30000], y_train[:30000], epochs=3, batch_size=BATCH_SIZE, verbose=0)
        personalized_models[client_name] = pers_model
    
    results = []
    for client_name in client_test_data.keys():
        X_test, y_test = client_test_data[client_name]
        X_ts = X_test.iloc[:20000] if len(X_test) > 20000 else X_test
        y_ts = y_test.iloc[:20000] if len(y_test) > 20000 else y_test
        y_pred = personalized_models[client_name].predict(X_ts, verbose=0).flatten()
        
        mae = mean_absolute_error(y_ts, y_pred)
        rmse = np.sqrt(mean_squared_error(y_ts, y_pred))
        mask = y_ts > 1.0
        mape = mean_absolute_percentage_error(y_ts[mask], y_pred[mask]) * 100 if mask.sum() > 0 else 0.0
        r2 = r2_score(y_ts, y_pred)
        results.append({'Client': client_name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
                        'MAPE (%)': round(mape, 2), 'R²': round(r2, 4)})
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # 7. Save Best Model
    best_client = df_results.loc[df_results['R²'].idxmax(), 'Client']
    best_r2 = df_results['R²'].max()
    best_model = personalized_models[best_client]
    model_filename = f'best_model_{dataset_name}_{best_client}_R2_{best_r2:.4f}.keras'
    best_model.save(model_filename)
    print(f"\n✅ Best model saved: {model_filename}")
    print(f"🏆 Best Client: {best_client} (R² = {best_r2:.4f})")
    
    return df_results, model_filename

# ==============================================================================
# Run on REFIT
# ==============================================================================
refit_path = 'data/REFIT/'
refit_files = ['CLEAN_House1.csv', 'CLEAN_House2.csv', 'CLEAN_House3.csv', 'CLEAN_House4.csv', 'CLEAN_House5.csv']
df_refit, model_refit = run_federated_learning(refit_path, refit_files, dataset_name='REFIT')

# ==============================================================================
# Run on UK-DALE
# ==============================================================================
ukdale_path = 'data/UK_DALE/'
ukdale_files = ['UK_DALE_Client_1_1min.csv', 'UK_DALE_Client_2_1min.csv', 
                'UK_DALE_Client_3_1min.csv', 'UK_DALE_Client_4_1min.csv']
df_ukdale, model_ukdale = run_federated_learning(ukdale_path, ukdale_files, dataset_name='UK-DALE')

# ==============================================================================
# Final Summary
# ==============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY: SAVED MODELS")
print("="*70)
print(f"📁 REFIT Model: {model_refit}")
print(f"📁 UK-DALE Model: {model_ukdale}")
print("\n✅ All deliverables are ready:")
print("   1. Technical Report (LaTeX PDF)")
print("   2. Complete Source Code (Python files)")
print("   3. Trained Best-Performing Models (2 .keras files)")