import pandas as pd  
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 
from imblearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

df=pd.read_csv('college_match.csv',low_memory=False)

print("\n" + "=" *70)
print("Head Of The Dataset: ")
print(df.head())
print("\n" + "=" *70)

print("\n" + "=" *70)
print(" Info Of The Dataset: ")
print(df.info())
print("\n" + "=" *70)

print("\n" + "=" *70)
print("Columns Of The Dataset: ")
print(df.columns.tolist())
print("\n" + "=" *70)

print("\n" + "=" *70)
print("Dtypes Of The Dataset: ")
print(df.dtypes)
print("\n" + "=" *70)

print("\n" + "=" *70)
print("Missing Values Of The Dataset: ")
print(df.isnull().sum())
print("\n" + "=" *70)

print("\n" + "=" *70)
print("Total Missing Values Of The Dataset: ")
print(df.isnull().sum().sum())
print("\n" + "=" *70)

print("\n" + "=" *70)
print("nPercentage Of The Missing Values Of The Dataset: ")
print(df.isnull().sum()/len(df)*100)
print("\n" + "=" *70)

print("\n" + "=" *70)
print("Duplicated Values Of The Dataset: ")
print(df.duplicated().sum())
print("\n" + "=" *70)

print("\n" + "=" *70)
print("Summary Of The Dataset: ")
print(df.describe())
print("\n" + "=" *70)

cat_cols=df.select_dtypes(include=['object','category']).columns.tolist()
for col in cat_cols:
    print("\n" + "=" *70)
    print("Categorical Analysis")
    print("\n" + "=" *70)
    print(df[col].value_counts(dropna=False))
    print(col, df[col].nunique(),df[col].unique()[:10])


numeric_col=df.select_dtypes(include=['number']).columns.tolist()
for c in numeric_col:
        print("\n" + "=" *70)
        print(f"Outlier Analysis: {c}")
        print("\n" + "=" *70)
        Q1=df[c].quantile(0.25)
        Q3=df[c].quantile(0.75)
        IQR=Q3-Q1
        outliers=((df[c] < Q1 - 1.5 *IQR) | df[c]>Q3 + 1.5 *IQR)  
        print(f"{c}: {outliers.sum()} outliers") 

numeric_col=df.select_dtypes(include=['number']).columns.tolist()
for k in numeric_col:
      print("\n" + "=" *70)
      print(f"Numeric Analysi: {k}")    
      print("\n" + "=" *70)
      print(
            f"Count  :{df[k].count()}\n"
            f"Mean   :{df[k].mean()}\n"
            f"Median :{df[k].median()}\n"
            f"Minimum :{df[k].min()}\n"
            f"Maximum :{df[k].max()}\n"
            f"Standard dev: {df[k].std()}"
      )


df=df.drop_duplicates()
print("duplicated:", df.duplicated().sum())

df["tuition_match"] = (
    df["TUITIONFEE_OUT"] <= 20000
)

df["state_match"] = (
    df["STABBR"] == "CA"
)

df["type_match"] = (
    df["CONTROL"] == 1
)

df["sat_match"] = (
    df["SAT_AVG"] >= 1100
)

df["act_match"] = (
    df["ACTCMMID"] >= 22
)

df["size_match"] = (
    df["UGDS"].between(5000, 15000)
)

labelenc=LabelEncoder()
df["tuition_match"]=labelenc.fit_transform(df["tuition_match"])
print(df["tuition_match"].value_counts())


x=df.drop(columns=[
    "STABBR",
    "CONTROL",
    "SAT_AVG",
    "ACTCMMID",
    "TUITIONFEE_OUT",
    "UGDS",
    "college_match"
])

y = df["college_match"]

print(y.value_counts())


numeric_cols=x.select_dtypes(include=['number']).columns.tolist()
cat_cols=x.select_dtypes(include=['category','object']).columns.tolist()

preprocessor=ColumnTransformer(transformers=[
      ("num",Pipeline([
            ("imput",SimpleImputer(strategy='median')),
            ("scaler",StandardScaler())
      ]),numeric_cols),

      ("cat", Pipeline([
            ("impute",SimpleImputer(strategy='most_frequent')),
            ("onehot",OneHotEncoder(handle_unknown='ignore'))
      ]),cat_cols)
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

models={
      "Linear": LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced'),
      "Svm": SVC(class_weight='balanced'),
      "RandomF": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
      )
}


results=[]

for name,model in models.items():
      
    
      pipeline=Pipeline([
      ("prepr",preprocessor),
      ("model",model)
])
      pipeline.fit(x_train,y_train)
      prediction=pipeline.predict(x_test)


      
      results.append({
      "Model": name,
      "Accuracy": accuracy_score(y_test,prediction),
      "Precision": precision_score(y_test,prediction),
      "Recall": recall_score(y_test,prediction),
      "F1_score": f1_score(y_test,prediction),
      "ROC": roc_auc_score(y_test,prediction),
})

results_df = pd.DataFrame(results)
print(results_df)    

best_model=Pipeline(steps=[
       ("prepr",preprocessor),
        ("model",SVC(class_weight='balanced',probability=True))
])

final=[]

best_model.fit(x_train,y_train)
joblib.dump(best_model,"college_match_model.pkl")

trained_pred=best_model.predict(x_train)
tested_pred=best_model.predict(x_test)
TRAINED_ACC=accuracy_score(y_train,trained_pred)
TESTED_ACC=accuracy_score(y_test,tested_pred)
print(
      f"TRAINED:  {TRAINED_ACC:.2f}\n"
      f"TESTED :  {TESTED_ACC: .2f}\n"
)
prediction=best_model.predict(x_test)

final.append({
    "Model": 'SVC',
    "Accuracy": accuracy_score(y_test,prediction),
    "Precision": precision_score(y_test,prediction,zero_division=0),
    "Recall": recall_score(y_test,prediction,zero_division=0),
    "F1_score": f1_score(y_test,prediction,zero_division=0),
    "ROC": roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1]),
      
})

dff=pd.DataFrame(final)
print(dff)


perm=permutation_importance(
      best_model,
      x_test,
      y_test,
      n_repeats=5,
      random_state=42
)

imp=pd.DataFrame({
      "feature": x_test.columns,
      "importance": perm.importances_mean
}).sort_values(by="importance",ascending=False)

top_cols=imp.head(10)
print(top_cols)



