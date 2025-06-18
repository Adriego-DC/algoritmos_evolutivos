import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar resultados generados por las 3 representaciones
binaria = np.load('algoritmos_evolutivos-s8_lab/resultado_binaria_modificada.npz', allow_pickle=True)
permutacional = np.load('algoritmos_evolutivos-s8_lab/resultado_permutacional_modificada.npz', allow_pickle=True)
real = np.load('algoritmos_evolutivos-s8_lab/resultado_real_modificada.npz', allow_pickle=True)

# Extraer fitness
fitness_binaria = binaria['fitness']
fitness_permutacional = permutacional['fitness']
fitness_real = real['fitness']

# Extraer notas agrupadas por examen
notas_binaria = binaria['notas'][()]          
notas_permutacional = permutacional['notas'][()]  
notas_real = real['notas'][()]  

print("======= DEPURACIÓN =======")
print("Tipo de notas_binaria:", type(notas_binaria))
print("Contenido de notas_binaria:", notas_binaria)
print("Tipo de notas_permutacional:", type(notas_permutacional))
print("Contenido de notas_permutacional:", notas_permutacional)
print("Tipo de notas_real:", type(notas_real))
print("Contenido de notas_real:", notas_real)
print("=========================")

# Gráfico de evolución del fitness
plt.figure(figsize=(12, 6))
plt.plot(fitness_binaria, label='Binaria')
plt.plot(fitness_permutacional, label='Permutacional')
plt.plot(fitness_real, label='Real')
plt.title('Evolución del Fitness por Generación')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Función para graficar histograma de notas por examen
def plot_histograma(notas, titulo):
    print(f"Graficando: {titulo} | Tipo de notas: {type(notas)} | Contenido: {notas}")
    plt.figure(figsize=(8, 4))
    for examen, notas_examen in notas.items():
        sns.histplot(notas_examen, kde=False, bins=range(0, 21), label=f'Examen {examen}', alpha=0.5)
    plt.title(titulo)
    plt.xlabel('Notas')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Histogramas de notas para cada representación
plot_histograma(notas_binaria, 'Distribución de Notas - Representación Binaria')
plot_histograma(notas_permutacional, 'Distribución de Notas - Representación Permutacional')
plot_histograma(notas_real, 'Distribución de Notas - Representación Real')

# Comparación de promedios de notas entre representaciones
plt.figure(figsize=(12, 6))
labels = ['A', 'B', 'C']
for metodo, notas in zip(['Binaria', 'Permutacional', 'Real'],
                         [notas_binaria, notas_permutacional, notas_real]):
    promedios = [np.mean(notas[examen]) for examen in labels]
    plt.plot(labels, promedios, marker='o', label=metodo)

plt.title('Comparación de Promedios de Notas por Examen entre Representaciones')
plt.xlabel('Examen')
plt.ylabel('Promedio de Notas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
