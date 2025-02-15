import pandas as pd

# Fayl yollarını təyin edirik
pritychenko_path = "/content/Pritychenko 2016_Quadrupole deformation_final.xlsx"
raman_path = "/content/Raman 2001_Quadrupole deformation data_final.xlsx"

# Excel fayllarını açırıq
xls_pritychenko = pd.ExcelFile(pritychenko_path)
xls_raman = pd.ExcelFile(raman_path)

# Mövcud sheet adlarını görmək üçün
print("Pritychenko Sheet Names:", xls_pritychenko.sheet_names)
print("Raman Sheet Names:", xls_raman.sheet_names)


# Hər iki faylı yükləyirik (əvvəlki addımdakı sheet adlarına uyğun olaraq)
df_pritychenko = pd.read_excel(xls_pritychenko, sheet_name="Sayfa2")
df_raman = pd.read_excel(xls_raman, sheet_name="Sayfa4")

# İlk 5 sətiri çap edirik
print("Pritychenko Dataset Preview:")
print(df_pritychenko.head())

print("\nRaman Dataset Preview:")
print(df_raman.head())

print("Pritychenko Columns:", df_pritychenko.columns)
print("Raman Columns:", df_raman.columns)


# Bütün lazımlı sütunları seçirik
columns_needed = ["Z", "N", "A", "E(keV)", "E_error(keV)", "B(E2) (e2b2)", "B(E2)_err (e2b2)",
                  "τ (ps)", "τ_error (ps)", "β2", "β2_error", "Q0(b)", "Q0(b)_error"]

df_pritychenko = df_pritychenko[columns_needed]
df_raman = df_raman[columns_needed]

# Sütun adlarını uyğunlaşdırırıq
df_raman.columns = df_pritychenko.columns

# Datasetləri A, Z, N dəyərlərinə görə birləşdiririk
df_combined = pd.merge(df_pritychenko, df_raman, on=["Z", "N", "A"], how="outer", suffixes=('_pri', '_ram'))

# Sütunları düzgün doldurmaq üçün ağıllı merge edirik
for col in columns_needed[3:]:  # İlk üç sütun A, Z, N olduğu üçün onları saxlayırıq
    col_pri = col + "_pri"
    col_ram = col + "_ram"
    
    if col_pri in df_combined.columns and col_ram in df_combined.columns:
        df_combined[col] = df_combined[col_pri].combine_first(df_combined[col_ram])  # Əgər Pritychenko-da boşdursa, Raman-dan götür
        df_combined.drop([col_pri, col_ram], axis=1, inplace=True)  # Lazımsız sütunları silirik

# NaN olan sətirlərin sayını yoxlayırıq
print("Boş xanaların sayı:")
print(df_combined.isnull().sum())

# İlk 5 sətiri göstəririk
print(df_combined.head())



import pandas as pd

# Google Colab üçün fayl yolları (faylları Colab-a yükləmisənsə, `content/` qovluğunda olmalıdır)
pritychenko_path = "/content/Pritychenko_2016.xlsx"  # Fayl adını öz faylınla dəyiş
raman_path = "/content/Raman_2001.xlsx"  # Fayl adını öz faylınla dəyiş

# Datasetləri yükləyirik
df_pritychenko = pd.read_excel(pritychenko_path, sheet_name="Sayfa2")
df_raman = pd.read_excel(raman_path, sheet_name="Sayfa4")

# Lazım olan sütunlar
columns_needed = ["Z", "N", "A", "E(keV)", "E_error(keV)", "B(E2) (e2b2)", "B(E2)_err (e2b2)",
                  "τ (ps)", "τ_error (ps)", "β2", "β2_error", "Q0(b)", "Q0(b)_error"]

df_pritychenko = df_pritychenko[columns_needed]
df_raman = df_raman[columns_needed]

# Sütun adlarını uyğunlaşdırırıq
df_raman.columns = df_pritychenko.columns

# Datasetləri A, Z, N dəyərlərinə görə birləşdiririk
df_combined = pd.merge(df_pritychenko, df_raman, on=["Z", "N", "A"], how="outer", suffixes=('_pri', '_ram'))

# **Boş olan sütunları ağıllı şəkildə doldururuq**
for col in columns_needed[3:]:  
    col_pri = col + "_pri"
    col_ram = col + "_ram"
    
    if col_pri in df_combined.columns and col_ram in df_combined.columns:
        df_combined[col] = df_combined[col_pri].combine_first(df_combined[col_ram])  # Əgər Pritychenko-da boşdursa, Raman-dan götür
        df_combined.drop([col_pri, col_ram], axis=1, inplace=True)  # Lazımsız sütunları silirik

# **Boş xanaları yoxlayırıq**
print("Boş xanaların sayı:")
print(df_combined.isnull().sum())

# **Faylı Google Colab-da `/content/` qovluğuna yazırıq**
output_path = "/content/combined_dataset.xlsx"
df_combined.to_excel(output_path, index=False)
print(f"Birləşdirilmiş dataset saxlanıldı: {output_path}")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# **Birləşdirilmiş datasetin faylını yükləyirik**
combined_path = "/content/combined_dataset.xlsx"  # Google Colab-da saxlanıb
df_combined = pd.read_excel(combined_path)

# **Proqnozlaşdıracağımız sütunlar** (yalnız β₂, Q₀ və B(E2))
columns_to_predict = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə edəcəyimiz giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# Boş xanaları Random Forest ilə doldurmaq üçün funksiya
def fill_missing_values(df, target_column, feature_cols):
    """
    Verilmiş sütundakı boş xanaları Random Forest modelindən istifadə edərək proqnozlaşdırır.
    """
    df_temp = df.copy()

    # Boş olmayan dəyərləri ayırırıq
    train_data = df_temp.dropna(subset=[target_column])

    # Əgər bütün sütun boşdursa, modeli işlətməyə ehtiyac yoxdur
    if train_data.shape[0] == 0:
        return df_temp

    # Hədəf dəyəri (Target) və giriş dəyişənləri (Features)
    X = train_data[feature_cols]
    y = train_data[target_column]

    # Boş olan sətirləri ayırırıq
    missing_data = df_temp[df_temp[target_column].isnull()]
    X_missing = missing_data[feature_cols]

    # Əgər boş hissədə giriş dəyişənləri qalmırsa, modeli işə salmırıq
    if X.shape[1] == 0 or X_missing.shape[0] == 0:
        return df_temp

    # Modeli qururuq və öyrədirik
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Boş olan hissələr üçün proqnoz alırıq
    df_temp.loc[df_temp[target_column].isnull(), target_column] = rf.predict(X_missing)

    return df_temp

# **β₂, Q₀ və B(E2) üçün proqnoz alırıq**
for column in columns_to_predict:
    print(f"Modelləşdirilir: {column}")
    df_combined = fill_missing_values(df_combined, column, feature_columns)

# **Final boş xanaları yoxlayırıq**
print("Final Boş Xanalar:")
print(df_combined.isnull().sum())

# **Proqnozlaşdırılmış dataset fayl kimi saxlanır**
df_combined.to_excel("/content/final_filled_deformation.xlsx", index=False)
print("Doldurulmuş dataset saxlanıldı: final_filled_deformation.xlsx")



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
