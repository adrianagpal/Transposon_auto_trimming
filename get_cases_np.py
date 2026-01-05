import numpy as np

# Cargar el archivo .npy
datos = np.load("dataset_autotrim2/species_labels.npy", allow_pickle=True)

# Guardar como archivo de texto
np.savetxt("dataset_autotrim2/labels_data.txt", datos, fmt="%s")

# Cargar el archivo .npy
datos = np.load("dataset_autotrim2/case_labels.npy")

# Guardar como archivo de texto
np.savetxt("dataset_autotrim2/case_labels.txt", datos, fmt="%s")

# Cargar el archivo .npy
datos = np.load("dataset_autotrim2/labels_data.npy")

# Guardar como archivo de texto
np.savetxt("dataset_autotrim2/labels.txt", datos, fmt="%s")

X_test = np.load("dataset_autotrim2/features_data.npy")

print("X_test dtype:", X_test.dtype)
print("X_test shape:", X_test.shape)
print("X_test min:", X_test.min())
print("X_test max:", X_test.max())
print("X_test mean:", X_test.mean())
print("NaNs in X_test:", np.isnan(X_test).sum())

i = 0
print("Sample", i)
print("Min per channel:", X_test[i].min(axis=(0,1)))
print("Max per channel:", X_test[i].max(axis=(0,1)))
print("Mean per channel:", X_test[i].mean(axis=(0,1)))

print("Patch of X_test[0, :5, :5, 0]:")
print(X_test[0, :5, :5, 0])
