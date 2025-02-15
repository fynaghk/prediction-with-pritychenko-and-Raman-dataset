import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Proqnozlaşdırılmış datasetin faylını yükləyirik**
final_filled_path = "/content/final_filled_deformation.xlsx"  # Əvvəlki nəticə
df_filled = pd.read_excel(final_filled_path)

# **Qiymətləndiriləcək sütunlar (β₂, Q₀, B(E2))**
columns_to_evaluate = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# Performans nəticələrini saxlamaq üçün boş siyahılar
mae_scores = {}
rmse_scores = {}
r2_scores = {}

# **Bütün sətirləri proqnozlaşdırmaq üçün yeni dataset**
df_predictions = df_filled.copy()

# **Hər sütun üçün Random Forest modeli qurub performansı ölçək**
for column in columns_to_evaluate:
    print(f"\n📊 Modelləşdirilir və Qiymətləndirilir: {column}")

    # **Boş olmayan dəyərləri ayırırıq (Model üçün Train Data)**
    train_data = df_filled.dropna(subset=[column])
    
    if train_data.shape[0] == 0:
        print(f"❌ {column} sütununda heç bir dəyər yoxdur, model qurulmayacaq.")
        continue

    X_train = train_data[feature_columns]
    y_train = train_data[column]

    # **Bütün dataset üçün proqnoz alacağıq**
    X_all = df_filled[feature_columns]

    # **Modeli qururuq və öyrədirik**
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # **Bütün dataset üçün proqnoz alırıq**
    y_pred = rf.predict(X_all)

    # **Yeni predict sütunu əlavə edirik**
    df_predictions[column + "_predict"] = y_pred

    # **Performans metriklərini hesablayırıq (yalnız orijinal məlumatlar üçün)**
    y_true = train_data[column]
    y_train_pred = rf.predict(X_train)

    mae = mean_absolute_error(y_true, y_train_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_train_pred))
    r2 = r2_score(y_true, y_train_pred)

    mae_scores[column] = mae
    rmse_scores[column] = rmse
    r2_scores[column] = r2

    print(f"📌 MAE: {mae:.4f}")
    print(f"📌 RMSE: {rmse:.4f}")
    print(f"📌 R²: {r2:.4f}")

# **Nəticələri göstəririk**
print("\n✅ **Model Performansı:**")
print("📌 MAE:", mae_scores)
print("📌 RMSE:", rmse_scores)
print("📌 R²:", r2_scores)

# **Final datasetin proqnozları ilə birlikdə saxlanması**
output_path = "/content/final_predictions.xlsx"
df_predictions.to_excel(output_path, index=False)
print(f"\n✅ Doldurulmuş dataset PROQNOZ sütunları ilə saxlanıldı: {output_path}")


# Daha aydın qrafiklər üçün məlumat sayını azaldırıq (nümunə götürmə)
import seaborn as sns
import matplotlib.pyplot as plt

# Stil təyin edirik
sns.set(style="whitegrid")

# Proqnozlaşdırılmış datasetin yüklənməsi
final_predictions_path = "/content/final_predictions.xlsx"
df_predictions = pd.read_excel(final_predictions_path)

# Proqnozlaşdırılan sütunlar
columns_to_plot = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# Qrafiklərin daha aydın görünməsi üçün nümunə götürək
df_sampled = df_predictions.sample(frac=0.3, random_state=42)  # 30% nümunə götürürük

# Qrafikləri daha sadə çəkmək
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    
    # Orijinal və proqnozlaşdırılmış dəyərləri göstəririk (nümunə götürülmüş)
    plt.scatter(df_sampled["A"], df_sampled[column], label=f"Original {column}", color="blue", marker="x", alpha=0.7, s=80)
    plt.scatter(df_sampled["A"], df_sampled[column + "_predict"], label=f"Predicted {column}", color="red", marker="x", alpha=0.7, s=80)
    
    plt.xlabel("Mass Number (A)", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.title(f"Original vs Predicted {column}", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.show()
