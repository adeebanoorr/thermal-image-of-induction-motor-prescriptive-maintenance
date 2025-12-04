#train_model.py
import os
import cv2
import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
CSV_PATH = "synthetic_maintenance_log.csv"
IMAGE_BASE_DIR = r"D:\projects\thermal_induction_motor\fault detection and severity prediction\IR-Motor-bmp"

# Create models directory
os.makedirs("models", exist_ok=True)

# Model and tool paths
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/fault_label_encoder.pkl"
ACTION_ENCODER_PATH = "models/action_label_encoder.pkl"
FAULT_MODEL_PATH = "models/fault_model.pkl"
SEVERITY_MODEL_PATH = "models/severity_model.pkl"
ACTION_MODEL_PATH = "models/action_model.pkl"
COST_MODEL_PATH = "models/cost_model.pkl"
DOWNTIME_MODEL_PATH = "models/downtime_model.pkl"

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Label mapping ===
def map_fault_label(label):
    label = label.lower()
    if "noload" in label or "no load" in label:
        return "Healthy"
    elif "rotor" in label:
        return "Rotor fault"
    elif "fan" in label:
        return "Cooling fan fault"
    elif any(x in label for x in ["a", "b", "c", "winding", "stator"]):
        return "Stator fault"
    else:
        return label

df['Fault_Type'] = df['Fault_Type'].apply(map_fault_label)

# === ResNet18 Feature Extractor ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(tensor)
    return features.cpu().numpy().reshape(-1)

# === Extract Features ===
features_list = []
fault_labels = []
severity_labels = []
action_labels = []
cost_values = []
downtime_values = []

print("🔍 Extracting features from images...")
for idx, row in df.iterrows():
    folder = row['Folder']
    image_path = os.path.join(IMAGE_BASE_DIR, folder, row['Image_Name'])
    if not os.path.exists(image_path):
        print(f"⚠️ Missing: {image_path}")
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to load: {image_path}")
        continue

    try:
        features = extract_features(img)
        features_list.append(features)
        fault_labels.append(row['Fault_Type'])
        severity_labels.append(row['Severity'])
        action_labels.append(row['ActionTaken'])
        cost_values.append(row['Cost'])
        downtime_values.append(row['Downtime_Days'])
        print(f"✅ Processed: {row['Image_Name']}")
    except Exception as e:
        print(f"❌ Error: {row['Image_Name']} → {e}")

# === To arrays ===
X = np.vstack(features_list)
y_fault = np.array(fault_labels)
y_severity = np.array(severity_labels)
y_action = np.array(action_labels)
y_cost = np.array(cost_values)
y_downtime = np.array(downtime_values)

# === Encode categorical labels ===
fault_encoder = LabelEncoder()
action_encoder = LabelEncoder()
y_fault_encoded = fault_encoder.fit_transform(y_fault)
y_action_encoded = action_encoder.fit_transform(y_action)

joblib.dump(fault_encoder, ENCODER_PATH)
joblib.dump(action_encoder, ACTION_ENCODER_PATH)
print("✅ Encoders saved")

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print("✅ Scaler saved")

# === Train/Test Split ===
X_train, X_test, y_fault_train, y_fault_test, y_sev_train, y_sev_test, \
y_action_train, y_action_test, y_cost_train, y_cost_test, y_down_train, y_down_test = train_test_split(
    X_scaled, y_fault_encoded, y_severity, y_action_encoded,
    y_cost, y_downtime, test_size=0.2, random_state=42, stratify=y_fault_encoded
)

print("\n📌 Train/Test Split Completed (80/20)\n")

# === Train Models ===
fault_model = RandomForestClassifier(n_estimators=100, random_state=42)
severity_model = RandomForestRegressor(n_estimators=100, random_state=42)
action_model = RandomForestClassifier(n_estimators=100, random_state=42)
cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
downtime_model = RandomForestRegressor(n_estimators=100, random_state=42)

fault_model.fit(X_train, y_fault_train)
severity_model.fit(X_train, y_sev_train)
action_model.fit(X_train, y_action_train)
cost_model.fit(X_train, y_cost_train)
downtime_model.fit(X_train, y_down_train)

# === Save Models ===
joblib.dump(fault_model, FAULT_MODEL_PATH)
joblib.dump(severity_model, SEVERITY_MODEL_PATH)
joblib.dump(action_model, ACTION_MODEL_PATH)
joblib.dump(cost_model, COST_MODEL_PATH)
joblib.dump(downtime_model, DOWNTIME_MODEL_PATH)

print("✅ All models trained and saved!\n")

# ============================================================
#                🔍 ACCURACY / EVALUATION METRICS
# ============================================================

print("\n===============================")
print("📊 MODEL PERFORMANCE METRICS")
print("===============================\n")

# === Fault Classification Metrics ===
y_pred_fault = fault_model.predict(X_test)
print("🔧 Fault Classification Accuracy:", accuracy_score(y_fault_test, y_pred_fault))
print("\n📌 Classification Report:")
print(classification_report(y_fault_test, y_pred_fault, target_names=fault_encoder.classes_))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_fault_test, y_pred_fault))

# === Action Classification Metrics ===
y_pred_action = action_model.predict(X_test)
print("\n🛠 Action Recommendation Accuracy:", accuracy_score(y_action_test, y_pred_action))
print("\n📌 Action Classification Report:")
print(classification_report(y_action_test, y_pred_action, target_names=action_encoder.classes_))

# === Severity Regression ===
y_pred_sev = severity_model.predict(X_test)
print("\n🔥 Severity RMSE:", np.sqrt(mean_squared_error(y_sev_test, y_pred_sev)))
print("🔥 Severity MAE:", mean_absolute_error(y_sev_test, y_pred_sev))
print("🔥 Severity R²:", r2_score(y_sev_test, y_pred_sev))

# === Cost Regression ===
y_pred_cost = cost_model.predict(X_test)
print("\n💰 Cost RMSE:", np.sqrt(mean_squared_error(y_cost_test, y_pred_cost)))
print("💰 Cost MAE:", mean_absolute_error(y_cost_test, y_pred_cost))
print("💰 Cost R²:", r2_score(y_cost_test, y_pred_cost))

# === Downtime Regression ===
y_pred_down = downtime_model.predict(X_test)
print("\n⏱ Downtime RMSE:", np.sqrt(mean_squared_error(y_down_test, y_pred_down)))
print("⏱ Downtime MAE:", mean_absolute_error(y_down_test, y_pred_down))
print("⏱ Downtime R²:", r2_score(y_down_test, y_pred_down))

print("\n✅ Evaluation complete!")

# === CONFUSION MATRIX PLOTS ===
def plot_confusion_matrix(cm, class_names, title, file_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"📁 Saved confusion matrix: {file_name}")


# === Fault classification CM ===
cm_fault = confusion_matrix(y_fault_test, y_pred_fault)
plot_confusion_matrix(
    cm_fault,
    fault_encoder.classes_,
    "Fault Classification Confusion Matrix",
    "confusion_matrix_fault.png"
)

# === Action recommendation CM ===
cm_action = confusion_matrix(y_action_test, y_pred_action)
plot_confusion_matrix(
    cm_action,
    action_encoder.classes_,
    "Action Recommendation Confusion Matrix",
    "confusion_matrix_action.png"
)