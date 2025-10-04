# pandas_visualstudio_demo.py
"""
Pandas mastery demo for Visual Studio / VS Code on macOS (Apple M4).
Covers:
 - create example CSV/JSON
 - Series with custom index
 - DataFrame creation
 - inspect (dtypes, head, tail, describe)
 - slicing (iloc, by columns)
 - boolean-array slicing and numeric range filtering
 - duplicated, nunique, drop_duplicates
 - pd.to_numeric, pd.to_datetime (with errors='coerce')
 - default values using .apply()
 - pipeline using .pipe() that prints diagnostics
 - .pipe() + functools.partial for thresholded filtering
Saves cleaned outputs to ./outputs/
"""

from pathlib import Path
from functools import partial
import pandas as pd

OUT = Path.cwd() / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) Create example data files (CSV and JSON)
# ----------------------------
csv_text = """id,name,age,signup_date,score,flag
1,Alice,25,2025-01-10,85.5,True
2,Bob,thirty,2025/02/15,90,False
3,Charlie,22,15-03-2025,NaN,True
4,,40,2025-04-01,78,True
5,Eve,,2025-05-05,88,False
6,Frank,29,2025-06-06,,True
6,Frank,29,2025-06-06,88,True
"""

csv_path = OUT / "example_users.csv"
csv_path.write_text(csv_text, encoding="utf-8")

# Also write JSON for demonstration (records)
df_example = pd.read_csv(csv_path)
json_path = OUT / "example_users.json"
df_example.to_json(json_path, orient="records", date_format="iso", force_ascii=False, indent=2)

print(f"Example CSV saved to: {csv_path}")
print(f"Example JSON saved to: {json_path}\n")

# ----------------------------
# 2) Define a Pandas Series with a custom index
# ----------------------------
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="example_series")
print("Series with custom index:")
print(s, "\n")

# ----------------------------
# 3) Create a Pandas DataFrame with specified columns (read CSV)
# ----------------------------
df = pd.read_csv(csv_path)
print("Initial DataFrame (read from CSV):")
print(df, "\n")

# ----------------------------
# 4) Inspect the DataFrame: dtypes, head, tail, describe
# ----------------------------
print("Dtypes:")
print(df.dtypes)
print("\nHead:")
print(df.head())
print("\nTail:")
print(df.tail())
print("\nDescribe (include='all'):")
print(df.describe(include="all"), "\n")

# ----------------------------
# 5) Data slicing by row position (.iloc) and by column name
# ----------------------------
print("Slicing rows by position (.iloc[1:4]):")
print(df.iloc[1:4], "\n")

print("Slicing columns by name (['name','age']):")
print(df[["name", "age"]], "\n")

# ----------------------------
# 6) Slice using boolean flags array and by a numeric range (age)
# ----------------------------
bool_flags = df["flag"].astype(bool).to_numpy()  # ensure boolean numpy array
print("Boolean flags (from 'flag'):")
print(bool_flags)
print("\nRows where flag==True:")
print(df[bool_flags], "\n")

# Safe conversion of 'age' to numeric and filter by range
df["age_numeric"] = pd.to_numeric(df["age"], errors="coerce")  # invalid -> NaN
age_filtered = df[(df["age_numeric"] >= 25) & (df["age_numeric"] <= 40)]
print("Rows with age between 25 and 40 (after safe numeric conversion):")
print(age_filtered[["id", "name", "age", "age_numeric"]], "\n")

# ----------------------------
# 7) duplicated, nunique, drop_duplicates
# ----------------------------
print("Count of unique 'id' (nunique):", df["id"].nunique())
print("\nDuplicated row mask (keep=False):")
print(df.duplicated(keep=False))
print("\nRows flagged as duplicates (keep=False):")
print(df[df.duplicated(keep=False)], "\n")

# Demonstrate drop_duplicates (by id + name)
df_no_dup = df.drop_duplicates(subset=["id", "name"], keep="first").reset_index(drop=True)
print("DataFrame after drop_duplicates(subset=['id','name']):")
print(df_no_dup, "\n")

# ----------------------------
# 8) Safe type conversions: pd.to_numeric and pd.to_datetime
# ----------------------------
df["score_numeric"] = pd.to_numeric(df["score"], errors="coerce")
df["signup_dt"] = pd.to_datetime(df["signup_date"], errors="coerce", dayfirst=False)
print("Converted columns (score_numeric and signup_dt):")
print(df[["score", "score_numeric", "signup_date", "signup_dt"]], "\n")

# ----------------------------
# 9) Set default values for missing data in a column using .apply()
# ----------------------------
def fill_name(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unknown"
    return x

df["name_filled"] = df["name"].apply(fill_name)
print("After filling missing 'name' using .apply():")
print(df[["name", "name_filled"]], "\n")

# ----------------------------
# 10) Implement data cleaning pipeline using .pipe() and print diagnostics
# ----------------------------
def convert_types_and_report(dframe):
    d = dframe.copy()
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["signup_date"] = pd.to_datetime(d["signup_date"], errors="coerce", dayfirst=False)
    print("--- Inside pipeline: dtypes after conversion ---")
    print(d.dtypes)
    print("\n--- Inside pipeline: null counts after conversion ---")
    print(d.isnull().sum())
    return d

print("Running cleaning pipeline via .pipe(convert_types_and_report):")
df_pipeline = df.pipe(convert_types_and_report)
print("\nPipeline result preview:")
print(df_pipeline.head(), "\n")

# ----------------------------
# 11) Use .pipe() with partial to apply threshold-based function
# ----------------------------
def filter_by_score_threshold(df_in, threshold=80):
    dfw = df_in.copy()
    dfw["score"] = pd.to_numeric(dfw["score"], errors="coerce")
    return dfw[dfw["score"] >= threshold]

filter_85 = partial(filter_by_score_threshold, threshold=85)
print("Applying .pipe() + partial to filter rows with score >= 85:")
high_score_rows = df_pipeline.pipe(filter_85)
print(high_score_rows[["id", "name", "score"]], "\n")

# ----------------------------
# 12) Final cleanup, save outputs
# ----------------------------
# Use filled names, converted numeric types, drop duplicate id+name
df_final = df_pipeline.copy()
df_final["name"] = df_final["name"].apply(fill_name)
df_final = df_final.drop_duplicates(subset=["id", "name"], keep="first").reset_index(drop=True)

csv_out = OUT / "cleaned_users.csv"
json_out = OUT / "cleaned_users.json"
series_out = OUT / "example_series.csv"

df_final.to_csv(csv_out, index=False)
df_final.to_json(json_out, orient="records", date_format="iso", force_ascii=False, indent=2)
s.to_csv(series_out, header=True)

print(f"Saved cleaned CSV: {csv_out}")
print(f"Saved cleaned JSON: {json_out}")
print(f"Saved example series CSV: {series_out}")
print("\nDemo finished. Check the outputs/ folder for generated files.")