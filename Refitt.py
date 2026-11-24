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

print("--- Project: Personalized Federated Learning (PFAW vs FedAvg) ---")

# ==============================================================================
# Step 1-4: Load Data (REFIT)
# ==============================================================================
data_path = 'data/REFIT/'
house_files = ['CLEAN_House1.csv', 'CLEAN_House2.csv', 'CLEAN_House3.csv', 'CLEAN_House4.csv', 'CLEAN_House5.csv']

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

# ==============================================================================
# Step 5-7: Advanced Feature Engineering
# ==============================================================================
def preprocess_client_data(df, dataset_type='REFIT'):
    df_processed = df.copy()
    
    if len(df_processed) > 400000:
        df_processed = df_processed.sample(n=400000, random_state=42).sort_index().reset_index(drop=True)
    
    if 'Issues' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Issues'])
    
    time_col = 'Time' if 'Time' in df_processed.columns else ('datetime' if 'datetime' in df_processed.columns else None)
    
    if dataset_type == 'REFIT':
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
    processed_df, target_col = preprocess_client_data(clients_data[client_name])
    client_processed_data[client_name] = processed_df
    target_columns[client_name] = target_col
    print(f"Processed {client_name} | Target: {target_col}")

# ==============================================================================
# Step 8: Train/Test Split
# ==============================================================================
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

# ==============================================================================
# Step 9: Build LSTM Model
# ==============================================================================
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

first_client = list(client_train_data.keys())[0]
input_dim = client_train_data[first_client][0].shape[1]
print(f"\nInput Dimension: {input_dim}")

# ==============================================================================
# Step 10: Aggregation Methods
# ==============================================================================
def get_weights(model): return model.get_weights()
def set_weights(model, weights): model.set_weights(weights)

def fed_avg_aggregate(global_model, client_models_list, client_weights=None):
    global_weights = get_weights(global_model)
    if client_weights is None:
        client_weights = np.ones(len(client_models_list)) / len(client_models_list)
    else:
        client_weights = np.array(client_weights) / np.sum(client_weights)
    new_weights = []
    for i in range(len(global_weights)):
        layer_weights = [get_weights(model)[i] for model in client_models_list]
        weighted_layer = np.zeros_like(layer_weights[0])
        for w, lw in zip(client_weights, layer_weights):
            weighted_layer += w * lw
        new_weights.append(weighted_layer)
    set_weights(global_model, new_weights)
    return global_model

def softmax_adaptive_weights(local_losses, temperature=10000.0):
    losses = np.array(local_losses)
    exp_weights = np.exp(-losses / temperature)
    return exp_weights / np.sum(exp_weights)

# ==============================================================================
# Step 11: FL Simulation (FedAvg vs PFAW)
# ==============================================================================
NUM_ROUNDS = 8
EPOCHS = 5
BATCH_SIZE = 512
SAMPLE_SIZE = 60000
MU = 0.01
early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

def train_local_fedprox(client_model, global_weights, X, y):
    client_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])
    local_weights = get_weights(client_model)
    new_weights = []
    for lw, gw in zip(local_weights, global_weights):
        new_weights.append(lw * (1 - MU) + gw * MU)
    set_weights(client_model, new_weights)
    return client_model

def run_fl_simulation(method='fedavg'):
    print(f"\n{'='*60}")
    print(f"FL Simulation - Method: {method.upper()}")
    print(f"{'='*60}")
    global_model = build_base_model(input_dim)
    round_losses = []
    
    for round_num in range(1, NUM_ROUNDS + 1):
        global_weights = get_weights(global_model)
        local_models = []
        local_mses = []
        
        for client_name in client_train_data.keys():
            client_model = build_base_model(input_dim)
            set_weights(client_model, global_weights)
            X_train, y_train = client_train_data[client_name]
            if len(X_train) > SAMPLE_SIZE:
                idx = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
                X_s, y_s = X_train.iloc[idx], y_train.iloc[idx]
            else:
                X_s, y_s = X_train, y_train
            
            if method == 'pfaw':
                client_model = train_local_fedprox(client_model, global_weights, X_s, y_s)
                loss = client_model.evaluate(X_s, y_s, verbose=0)[0]
            else:
                history = client_model.fit(X_s, y_s, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])
                loss = history.history['loss'][-1]
            
            local_models.append(client_model)
            local_mses.append(loss)
        
        if method == 'fedavg':
            global_model = fed_avg_aggregate(global_model, local_models, client_weights=None)
        elif method == 'pfaw':
            weights = softmax_adaptive_weights(local_mses, temperature=10000.0)
            global_model = fed_avg_aggregate(global_model, local_models, client_weights=weights)
        
        avg_loss = np.mean(local_mses)
        round_losses.append(avg_loss)
        print(f"  Round {round_num}/{NUM_ROUNDS} | Avg Loss: {avg_loss:.2f}")
    
    return global_model, round_losses

# Run FedAvg (Baseline) and PFAW (Proposed)
global_fedavg, losses_fedavg = run_fl_simulation(method='fedavg')
global_pfaw, losses_pfaw = run_fl_simulation(method='pfaw')

# ==============================================================================
# Step 12: Personalization & Evaluation
# ==============================================================================
def personalize_and_evaluate(global_model, method_name):
    print(f"\n--- Evaluation ({method_name}) ---")
    personalized_models = {}
    for client_name in client_train_data.keys():
        pers_model = build_base_model(input_dim)
        set_weights(pers_model, get_weights(global_model))
        X_train, y_train = client_train_data[client_name]
        pers_model.fit(X_train[:30000], y_train[:30000], epochs=3, batch_size=BATCH_SIZE, verbose=0)
        personalized_models[client_name] = pers_model
    
    results = []
    all_preds = {}
    for client_name in client_test_data.keys():
        X_test, y_test = client_test_data[client_name]
        X_ts = X_test.iloc[:20000] if len(X_test) > 20000 else X_test
        y_ts = y_test.iloc[:20000] if len(y_test) > 20000 else y_test
        y_pred = personalized_models[client_name].predict(X_ts, verbose=0).flatten()
        all_preds[client_name] = (y_ts.values, y_pred)
        
        mae = mean_absolute_error(y_ts, y_pred)
        rmse = np.sqrt(mean_squared_error(y_ts, y_pred))
        mask = y_ts > 1.0
        mape = mean_absolute_percentage_error(y_ts[mask], y_pred[mask]) * 100 if mask.sum() > 0 else 0.0
        r2 = r2_score(y_ts, y_pred)
        results.append({'Client': client_name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
                        'MAPE (%)': round(mape, 2), 'R²': round(r2, 4)})
    
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res, all_preds

df_fedavg, preds_fedavg = personalize_and_evaluate(global_fedavg, "FedAvg (Baseline)")
df_pfaw, preds_pfaw = personalize_and_evaluate(global_pfaw, "PFAW (Proposed)")

# ==============================================================================
# Step 13: Comparison
# ==============================================================================
print("\n" + "="*60)
print("FINAL COMPARISON: FedAvg (Baseline) vs PFAW (Proposed)")
print("="*60)
comparison = pd.DataFrame({
    'Client': df_fedavg['Client'],
    'FedAvg R²': df_fedavg['R²'],
    'PFAW R²': df_pfaw['R²'],
    'Improvement': (df_pfaw['R²'] - df_fedavg['R²']).round(4)
})
print(comparison.to_string(index=False))
print(f"\nAverage R² Improvement: {comparison['Improvement'].mean():.4f}")

# ==============================================================================
# Step 14: Visualizations
# ==============================================================================
# Plot 1: Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_ROUNDS+1), losses_fedavg, 'bo-', label='FedAvg (Baseline)')
plt.plot(range(1, NUM_ROUNDS+1), losses_pfaw, 'ro-', label='PFAW (Proposed)')
plt.xlabel('Federated Round')
plt.ylabel('Average Local Loss (MSE)')
plt.title('Convergence: FedAvg vs PFAW')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curves_pfaw.png', dpi=150)
plt.close()

# Plot 2: R² per Client
plt.figure(figsize=(10, 5))
x = np.arange(len(df_fedavg))
width = 0.35
plt.bar(x - width/2, df_fedavg['R²'], width, label='FedAvg', color='steelblue')
plt.bar(x + width/2, df_pfaw['R²'], width, label='PFAW', color='darkorange')
plt.xlabel('Client')
plt.ylabel('R² Score')
plt.title('R² per Client: FedAvg vs PFAW')
plt.xticks(x, df_fedavg['Client'])
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('r2_comparison_pfaw.png', dpi=150)
plt.close()

# Plot 3: Prediction vs Actual (Client 1)
plt.figure(figsize=(14, 5))
actual, predicted = preds_pfaw['Client_1']
plt.plot(actual[:200], label='Actual', color='blue', alpha=0.7)
plt.plot(predicted[:200], label='Predicted (PFAW)', color='red', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Aggregate Power (W)')
plt.title('Prediction vs Actual - Client 1 (PFAW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_pfaw.png', dpi=150)
plt.close()

print("\n✅ Project completed!")
print("📊 Plots saved: loss_curves_pfaw.png, r2_comparison_pfaw.png, prediction_pfaw.png")