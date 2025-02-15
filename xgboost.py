# XGBoost modelindən istifadə edərək boş xanaları proqnozlaşdırmaq və model performansını qiymətləndirmək
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Proqnozlaşdırılmış datasetin faylını yükləyirik**
final_filled_path = "/content/final_filled_deformation.xlsx"  # Əvvəlki nəticə
df_filled = pd.read_excel(final_filled_path)

# **Proqnozlaşdıracağımız sütunlar (yalnız β₂, Q₀ və B(E2))**
columns_to_evaluate = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# Performans nəticələrini saxlamaq üçün boş siyahılar
mae_scores = {}
rmse_scores = {}
r2_scores = {}

# **Bütün sətirləri proqnozlaşdırmaq üçün yeni dataset**
df_predictions_xgb = df_filled.copy()

# **Hər sütun üçün XGBoost modeli qurub performansı ölçək**
for column in columns_to_evaluate:
    print(f"\n📊 XGBoost Modelləşdirilir və Qiymətləndirilir: {column}")

    # **Boş olmayan dəyərləri ayırırıq (Model üçün Train Data)**
    train_data = df_filled.dropna(subset=[column])
    
    if train_data.shape[0] == 0:
        print(f"❌ {column} sütununda heç bir dəyər yoxdur, model qurulmayacaq.")
        continue

    X_train = train_data[feature_columns]
    y_train = train_data[column]

    # **Bütün dataset üçün proqnoz alacağıq**
    X_all = df_filled[feature_columns]

    # **XGBoost Modelini qururuq və öyrədirik**
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    # **Bütün dataset üçün proqnoz alırıq**
    y_pred = xgb_model.predict(X_all)

    # **Yeni predict sütunu əlavə edirik**
    df_predictions_xgb[column + "_predict_xgb"] = y_pred

    # **Performans metriklərini hesablayırıq (yalnız orijinal məlumatlar üçün)**
    y_true = train_data[column]
    y_train_pred = xgb_model.predict(X_train)

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
print("\n✅ **XGBoost Model Performansı:**")
print("📌 MAE:", mae_scores)
print("📌 RMSE:", rmse_scores)
print("📌 R²:", r2_scores)

# **Final datasetin proqnozları ilə birlikdə saxlanması**
output_path = "/content/final_predictions_xgb.xlsx"
df_predictions_xgb.to_excel(output_path, index=False)
print(f"\n✅ Doldurulmuş dataset XGBoost PROQNOZ sütunları ilə saxlanıldı: {output_path}")


# **Proqnozlaşdırılmış datasetin faylını yükləyirik**
final_filled_path = "/content/final_filled_deformation.xlsx"  # Əvvəlki nəticə
df_filled = pd.read_excel(final_filled_path)

# **Proqnozlaşdıracağımız sütunlar (yalnız β₂, Q₀ və B(E2))**
columns_to_predict = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# **XGBoost modeli ilə boş xanaları proqnozlaşdırmaq üçün funksiya**
def predict_with_xgb(df, target_column, feature_cols):
    """
    Bütün dataset üçün XGBoost modeli ilə proqnozlaşdırma aparır və predict sütunu əlavə edir.
    """
    df_temp = df.copy()

    # Boş olmayan dəyərləri ayırırıq
    train_data = df_temp.dropna(subset=[target_column])

    # Əgər sütun tamamilə boşdursa, modeli işlətmirik
    if train_data.shape[0] == 0:
        return df_temp

    # Hədəf dəyəri (Target) və giriş dəyişənləri (Features)
    X_train = train_data[feature_cols]
    y_train = train_data[target_column]

    # Bütün dataset üçün proqnoz alacağıq
    X_all = df_temp[feature_cols]

    # **XGBoost Modelini qururuq və öyrədirik**
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Bütün dataset üçün proqnoz alırıq
    df_temp[target_column + "_predict_xgb"] = xgb_model.predict(X_all)

    return df_temp

# **Hər sütun üçün predict sütunu əlavə edirik**
for column in columns_to_predict:
    print(f"📊 Proqnozlaşdırılır: {column}")
    df_filled = predict_with_xgb(df_filled, column, feature_columns)

# **Proqnozlaşdırılmış dataset fayl kimi saxlanır**
output_path_xgb = "/content/final_predictions_xgb.xlsx"
df_filled.to_excel(output_path_xgb, index=False)
print(f"\n✅ Yeni dataset XGBoost PROQNOZ sütunları ilə saxlanıldı: {output_path_xgb}")


# Lazım olan kitabxanaları yükləyirik
import pandas as pd
import numpy as np
import xgboost as xgb

# **Proqnozlaşdırılmış datasetin faylını yükləyirik**
final_filled_path = "/content/final_filled_deformation.xlsx"  # Əvvəlki nəticə
df_filled = pd.read_excel(final_filled_path)

# **Proqnozlaşdıracağımız sütunlar (yalnız β₂, Q₀ və B(E2))**
columns_to_predict = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# **XGBoost modeli ilə boş xanaları proqnozlaşdırmaq üçün funksiya**
def fill_missing_values_xgb(df, target_column, feature_cols):
    """
    Verilmiş sütundakı boş xanaları XGBoost modelindən istifadə edərək proqnozlaşdırır.
    """
    df_temp = df.copy()

    # Boş olmayan dəyərləri ayırırıq
    train_data = df_temp.dropna(subset=[target_column])

    # Əgər sütun tamamilə boşdursa, modeli işlətmirik
    if train_data.shape[0] == 0:
        return df_temp

    # Hədəf dəyəri (Target) və giriş dəyişənləri (Features)
    X_train = train_data[feature_cols]
    y_train = train_data[target_column]

    # Boş olan sətirləri ayırırıq
    missing_data = df_temp[df_temp[target_column].isnull()]
    X_missing = missing_data[feature_cols]

    # Əgər boş hissədə giriş dəyişənləri yoxdursa, modeli işə salmırıq
    if X_train.shape[1] == 0 or X_missing.shape[0] == 0:
        return df_temp

    # **XGBoost Modelini qururuq və öyrədirik**
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Boş olan hissələr üçün proqnoz alırıq
    df_temp.loc[df_temp[target_column].isnull(), target_column] = xgb_model.predict(X_missing)

    return df_temp

# **β₂, Q₀ və B(E2) sütunlarındakı boş xanaları XGBoost ilə proqnozlaşdırırıq**
for column in columns_to_predict:
    print(f"📊 XGBoost Modelləşdirilir: {column}")
    df_filled = fill_missing_values_xgb(df_filled, column, feature_columns)

# **Final boş xanaları yoxlayırıq**
print("✅ Final Boş Xanalar:")
print(df_filled.isnull().sum())

# **Proqnozlaşdırılmış dataset fayl kimi saxlanır**
output_path_xgb = "/content/final_filled_xgb.xlsx"
df_filled.to_excel(output_path_xgb, index=False)
print(f"\n✅ Doldurulmuş dataset XGBoost modeli ilə saxlanıldı: {output_path_xgb}")
