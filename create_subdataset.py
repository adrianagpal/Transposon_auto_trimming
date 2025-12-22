import numpy as np
import pandas as pd

# -------------------------------
# 1. Cargar datos
# -------------------------------
X2 = np.load("dataset_30000/features_data.npy")
labels = np.load("dataset_30000/labels_data.npy", allow_pickle=True)
cases = np.load("dataset_30000/case_labels.npy", allow_pickle=True)
species2 = np.load("dataset_30000/species_labels.npy", allow_pickle=True)  # vector de especies
species1 = np.load("new_dataset/species_labels.npy", allow_pickle=True)  # vector de referencia

# -------------------------------
# 2. Recortar X2 a la longitud de species2
# -------------------------------
X2_clean = X2[:len(species2)]
labels = labels[:len(species2)]
cases = cases[:len(species2)]
species2_clean = species2  # ya esta alineado

print("X2 original:", X2.shape)
print("labels:", labels.shape)
print("cases:", cases.shape)
print("X2 recortado:", X2_clean.shape)
print("species2:", len(species2_clean))

# -------------------------------
# 3. Normalizar nombres de especies
# -------------------------------
species1_norm = pd.Series(species1).str.strip().str.lower().values
species2_norm = pd.Series(species2_clean).str.strip().str.lower().values

# -------------------------------
# 4. Crear mascara: especies de species2 que NO estan en species1
# -------------------------------
mask_species = ~np.isin(species2_norm, species1_norm)

# -------------------------------
# 5. Aplicar mascara para filtrar
# -------------------------------
X2_small = X2_clean[mask_species]
species2_small = species2_clean[mask_species]
labels_small = labels[mask_species]
cases_small = cases[mask_species]

# Convertir a Series para limpiar
species1_series = pd.Series(species1).dropna().astype(str).str.strip().str.lower()
species2_series = pd.Series(species2_clean).dropna().astype(str).str.strip().str.lower()

# Ver especies unicas
print("Especies finales en species1:", species1_series.unique())
print("Especies finales en species2:", species2_series.unique())

print("X2 final:", X2_small.shape)
print("species2 final:", len(species2_small))
print("Especies finales:", np.unique(species2_small))


# -------------------------------
# 6. Guardar resultados
# -------------------------------
np.save("features_data.npy", X2_small)
np.save("species_labels.npy", species2_small)
np.save("labels_data.npy", labels_small)
np.save("case_labels.npy", cases_small)
