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


