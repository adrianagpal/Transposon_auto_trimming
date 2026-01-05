# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Cargar archivo NumPy
# -----------------------------
# Cargar species.npy
species_array = np.load("dataset_30000/species_labels.npy", allow_pickle=True)

# Eliminar None y valores vacios
species_clean = [s for s in species_array if s is not None and str(s).strip() != ""]

# Convertir a unicos
species_unique = np.unique(species_clean)

# Crear DataFrame
species_df = pd.DataFrame(species_unique, columns=["Species"])

# Limpieza: quitar espacios y reemplazar "_" por " "
species_df["Species"] = (
    species_df["Species"]
    .astype(str)
    .str.strip()
    .str.replace("_", " ")
)

species_df.to_csv("species_unique.csv", index=False)

# -----------------------------
# 2. Cargar el Excel con taxonomia
# -----------------------------
taxonomy_df = pd.read_csv("InpactorDB2_release_v.2.0_beta.1.6.1.csv", sep=";")

# Limpiar nombres de columna: quitar espacios y caracteres extra
taxonomy_df.columns = taxonomy_df.columns.str.strip()

# Opcional: normalizar mayusculas/minusculas
taxonomy_df.columns = taxonomy_df.columns.str.capitalize()

# Verificar que 'Species' exista
print(taxonomy_df.columns.tolist())
print(taxonomy_df[['Species']].head())

# Limpieza
taxonomy_df["Species"] = taxonomy_df["Species"].astype(str).str.strip()

# -----------------------------
# 3. Unir las especies con la tabla completa
# -----------------------------
merged = species_df.merge(taxonomy_df, on="Species", how="left")

print("\nVista previa del merge:")
print(merged.head())

# -----------------------------
# 4. Grafico: numero de especies por Familia
# -----------------------------
kingdom_counts = merged["Phylum"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(6, 4))

color = "#ADD8E6"

# Grafico de barras vertical
bars = plt.bar(kingdom_counts.index, kingdom_counts.values, color=color, edgecolor='black', width=0.7)

plt.ylim(0, max(kingdom_counts) * 1.1)  

# Anadir los valores encima de cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,  # posicion horizontal
        height + max(kingdom_counts)*0.03, # un poco por encima de la barra
        str(height), 
        ha='center', 
        va='bottom',
        fontsize=10
    )

# Etiquetas y titulo
plt.xlabel("Phylum", fontsize=11)
plt.ylabel("N\u00FAmero de especies", fontsize=11)
plt.title("N\u00FAmero de especies por Phylum", fontsize=12, weight='bold')

# Separar un poco las etiquetas del eje x para que no se solapen
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()  # ajusta margenes automaticamente

# Guardar y mostrar
plt.savefig("grafico_familias_vertical.png", dpi=300, bbox_inches="tight")

# -----------------------------
# 5. (Opcional) Grafico por Orden
# -----------------------------
# order_counts = merged["Order"].value_counts().sort_values(ascending=False)
# plt.figure(figsize=(10, 5))
# order_counts.plot(kind="bar")
# plt.title("Numero de especies por Orden")
# plt.show()
