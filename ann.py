import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Dataseti yükləyirik**
final_filled_path = "/content/combined_dataset.xlsx"
df_filled = pd.read_excel(final_filled_path)

# **Proqnozlaşdıracağımız sütunlar**
columns_to_predict = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# **NaN dəyərləri doldurmaq üçün SimpleImputer istifadə edirik (median)**
imputer = SimpleImputer(strategy="median")

# Giriş dəyişənlərini NaN dəyərlərsiz əldə edirik
df_filled[feature_columns] = imputer.fit_transform(df_filled[feature_columns])

# **Feature-ları miqyaslandırırıq (StandardScaler)**
scaler = StandardScaler()
df_filled[feature_columns] = scaler.fit_transform(df_filled[feature_columns])

# **ANN modeli üçün boş xanaları proqnozlaşdırmaq üçün funksiya**
def fill_missing_values_ann(df, target_column, feature_cols):
    """
    Verilmiş sütundakı boş xanaları ANN modelindən istifadə edərək proqnozlaşdırır.
    """
    df_temp = df.copy()

    # Boş olmayan dəyərləri ayırırıq
    train_data = df_temp.dropna(subset=[target_column])

    # Əgər sütun tamamilə boşdursa, modeli işlətmirik
    if train_data.shape[0] == 0:
        print(f"❌ {target_column} sütunu tamamilə boşdur. Model işlədilmədi.")
        return df_temp

    # Hədəf dəyəri və giriş dəyişənləri
    X_train = train_data[feature_cols].values
    y_train = train_data[target_column].values

    # Boş olan sətirləri ayırırıq
    missing_data = df_temp[df_temp[target_column].isnull()]
    X_missing = missing_data[feature_cols].values

    # Əgər boş hissədə giriş dəyişənləri yoxdursa, modeli işə salmırıq
    if X_train.shape[1] == 0 or X_missing.shape[0] == 0:
        print(f"❌ {target_column} sütununda giriş məlumatları yoxdur.")
        return df_temp

    # **ANN Modelini qururuq**
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # **Modeli öyrədirik**
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    # **Boş olan hissələr üçün proqnoz alırıq**
    df_temp.loc[df_temp[target_column].isnull(), target_column] = model.predict(X_missing).flatten()

    return df_temp

# **β₂, Q₀ və B(E2) sütunlarındakı boş xanaları ANN ilə proqnozlaşdırırıq**
for column in columns_to_predict:
    print(f"📊 ANN Modelləşdirilir: {column}")
    df_filled = fill_missing_values_ann(df_filled, column, feature_columns)

# **Final boş xanaları yoxlayırıq**
print("✅ Final Boş Xanalar:")
print(df_filled.isnull().sum())

# **Proqnozlaşdırılmış dataset fayl kimi saxlanır**
output_path_ann = "/content/final_filled_ann.xlsx"
df_filled.to_excel(output_path_ann, index=False)
print(f"\n✅ Doldurulmuş dataset ANN modeli ilə saxlanıldı: {output_path_ann}")

# **Modelin MAE, RMSE və R² Dəyərlərini Hesablamaq**
mae_scores_ann = {}
rmse_scores_ann = {}
r2_scores_ann = {}

for column in columns_to_predict:
    y_true = df_filled[column].dropna()
    y_pred = df_filled.loc[y_true.index, column]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mae_scores_ann[column] = mae
    rmse_scores_ann[column] = rmse
    r2_scores_ann[column] = r2

# **Model Performansını Çap Edirik**
print("\n✅ ANN Model Performansı:")
print("📌 MAE:", mae_scores_ann)
print("📌 RMSE:", rmse_scores_ann)
print("📌 R²:", r2_scores_ann)




# Lazımi kitabxanaları yükləyirik
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# **Dataseti yükləyirik**
final_filled_path = "/content/final_filled_deformation.xlsx"
df_filled = pd.read_excel(final_filled_path)

# **Proqnozlaşdıracağımız sütunlar**
columns_to_predict = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# **NaN dəyərləri doldurmaq üçün SimpleImputer istifadə edirik (median)**
imputer = SimpleImputer(strategy="median")
df_filled[feature_columns] = imputer.fit_transform(df_filled[feature_columns])

# **Feature-ları miqyaslandırırıq (StandardScaler)**
scaler = StandardScaler()
df_filled[feature_columns] = scaler.fit_transform(df_filled[feature_columns])

# **ANN modeli ilə proqnoz sütunları əlavə edən funksiya**
def predict_with_ann(df, target_column, feature_cols):
    """
    Bütün dataset üçün ANN modelindən istifadə edərək proqnozlaşdırma aparır və predict sütunu əlavə edir.
    """
    df_temp = df.copy()

    # Mövcud dəyərləri ayırırıq
    train_data = df_temp.dropna(subset=[target_column])

    # Əgər sütun tamamilə boşdursa, modeli işlətmirik
    if train_data.shape[0] == 0:
        print(f"{target_column} sütunu tamamilə boşdur. Model işlədilmədi.")
        return df_temp

    # Hədəf dəyəri və giriş dəyişənləri
    X_train = train_data[feature_cols].values
    y_train = train_data[target_column].values

    X_all = df_temp[feature_cols].values  # Bütün dataset üçün giriş dəyişənləri

    # **ANN Modelini qururuq**
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # **Modeli öyrədirik**
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    # **Bütün dataset üçün proqnoz alırıq**
    df_temp[target_column + "_predict_ann"] = model.predict(X_all).flatten()

    return df_temp

# **Bütün dataset üçün proqnoz alırıq**
for column in columns_to_predict:
    print(f"ANN Modelləşdirilir: {column}")
    df_filled = predict_with_ann(df_filled, column, feature_columns)

# **Proqnozlaşdırılmış dataset fayl kimi saxlanır**
output_path_ann_predict = "/content/final_predictions_ann.xlsx"
df_filled.to_excel(output_path_ann_predict, index=False)
print(f"\nProqnozlaşdırılmış dataset ANN modeli ilə saxlanıldı: {output_path_ann_predict}")

# **Seaborn Qrafikləri Çəkmək (30% nümunə götürərək)**
sns.set(style="whitegrid")

df_sampled = df_filled.sample(frac=0.3, random_state=42)  # Datasetin 30%-ni götürürük

for column in columns_to_predict:
    predict_column = column + "_predict_ann"

    plt.figure(figsize=(10, 6))
    
    # Orijinal və proqnozlaşdırılmış dəyərləri göstəririk (x markerləri ilə)
    sns.scatterplot(x=df_sampled["A"], y=df_sampled[column], label=f"Original {column}", color="blue", marker="x", alpha=0.7, s=80)
    sns.scatterplot(x=df_sampled["A"], y=df_sampled[predict_column], label=f"Predicted {column}", color="red", marker="x", alpha=0.7, s=80)
    
    plt.xlabel("Mass Number (A)", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.title(f"Original vs Predicted {column} (ANN)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.show()

