from tensorflow.keras.models import load_model

# Load REFIT model
refit_model = load_model(
    "best_model_REFIT_Client_5_R2_0.8270.keras"
)

# Load UK-DALE model
ukdale_model = load_model(
    "best_model_UK-DALE_Client_2_R2_0.8325.keras"
)

# Display REFIT model architecture
print("\n========== REFIT MODEL ==========")
refit_model.summary()

# Display UK-DALE model architecture
print("\n========== UK-DALE MODEL ==========")
ukdale_model.summary()