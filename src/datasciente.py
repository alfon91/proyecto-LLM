import pandas as pd
import os

df = pd.read_csv('data\dataframe_extraido.csv')

print(df.to_string())

columnas_numericas = ['base_imponible', 'iva', 'importe_iva', 'total']
for col in columnas_numericas:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('€', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
df.to_excel("data/facturas_analisis.xlsx", index=False, engine="openpyxl")
print("💾 Archivo Excel guardado correctamente sin errores de formato.")

