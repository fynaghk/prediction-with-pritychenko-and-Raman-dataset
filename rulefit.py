# RuleFit modelini istifadə edərək bütün dataset üçün proqnozlar almaq
!pip install rulefit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rulefit import RuleFit
from sklearn.impute import SimpleImputer

# **Dataseti yükləyirik**
final_filled_path = "/content/final_filled_deformation.xlsx"
df_filled = pd.read_excel(final_filled_path)

# **Proqnozlaşdıracağımız sütunlar**
columns_to_predict = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# **Model üçün istifadə ediləcək giriş dəyişənləri (features)**
feature_columns = ["Z", "N", "A", "E(keV)", "E_error(keV)", "τ (ps)", "τ_error (ps)"]

# **NaN dəyərləri doldurmaq üçün SimpleImputer istifadə edirik (mean və ya median)**
imputer = SimpleImputer(strategy="median")  # Alternativ: "mean"

# Giriş dəyişənlərini NaN dəyərlərsiz əldə edirik
df_filled[feature_columns] = imputer.fit_transform(df_filled[feature_columns])

# **RuleFit modeli ilə proqnoz almaq üçün funksiya**
def predict_with_rulefit(df, target_column, feature_cols):
    """
    Verilmiş sütun üçün bütün datasetdə RuleFit modelindən istifadə edərək proqnozlaşdırma aparır.
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

    # **RuleFit Modelini qururuq və öyrədirik**
    rulefit_model = RuleFit(tree_size=4, sample_fract=0.75, max_rules=200)
    rulefit_model.fit(X_train, y_train)

    # **Bütün dataset üçün proqnoz alırıq**
    df_temp[target_column + "_predict_rulefit"] = rulefit_model.predict(X_all)

    return df_temp

# **Bütün dataset üçün proqnoz alırıq**
for column in columns_to_predict:
    print(f"RuleFit Modelləşdirilir: {column}")
    df_filled = predict_with_rulefit(df_filled, column, feature_columns)

# **Final boş xanaları yoxlayırıq**
print("Final Boş Xanalar:")
print(df_filled.isnull().sum())

# **Proqnozlaşdırılmış dataset fayl kimi saxlanır**
output_path_rulefit = "/content/final_predictions_rulefit.xlsx"
df_filled.to_excel(output_path_rulefit, index=False)
print(f"\nProqnozlaşdırılmış dataset RuleFit modeli ilə saxlanıldı: {output_path_rulefit}")

# **Seaborn Qrafikləri Çəkmək (30% nümunə götürərək)**
sns.set(style="whitegrid")

df_sampled = df_filled.sample(frac=0.3, random_state=42)  # Datasetin 30%-ni götürürük

for column in columns_to_predict:
    predict_column = column + "_predict_rulefit"

    plt.figure(figsize=(10, 6))
    
    # Orijinal və proqnozlaşdırılmış dəyərləri göstəririk (x markerləri ilə)
    sns.scatterplot(x=df_sampled["A"], y=df_sampled[column], label=f"Original {column}", color="blue", marker="x", alpha=0.7, s=80)
    sns.scatterplot(x=df_sampled["A"], y=df_sampled[predict_column], label=f"Predicted {column}", color="red", marker="x", alpha=0.7, s=80)
    
    plt.xlabel("Mass Number (A)", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.title(f"Original vs Predicted {column} (RuleFit)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.show()




# MAE, RMSE və R² dəyərlərini hesablamaq üçün lazımi kitabxanaları əlavə edirik
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Performans nəticələrini saxlamaq üçün boş siyahılar
mae_scores_rulefit = {}
rmse_scores_rulefit = {}
r2_scores_rulefit = {}

# **Hər sütun üçün performans metriklərini hesablayırıq**
for column in columns_to_predict:
    predict_column = column + "_predict_rulefit"

    # Əgər proqnoz sütunu əlavə olunubsa, dəyərləri hesablayaq
    if predict_column in df_filled.columns:
        y_true = df_filled[column].dropna()
        y_pred = df_filled.loc[y_true.index, predict_column]  # Yalnız mövcud dəyərlərlə müqayisə edirik

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mae_scores_rulefit[column] = mae
        rmse_scores_rulefit[column] = rmse
        r2_scores_rulefit[column] = r2

# **Model Performansını Çap Edirik**
print("\nRuleFit Model Performansı:")
print("MAE:", mae_scores_rulefit)
print("RMSE:", rmse_scores_rulefit)
print("R²:", r2_scores_rulefit)

