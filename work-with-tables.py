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
