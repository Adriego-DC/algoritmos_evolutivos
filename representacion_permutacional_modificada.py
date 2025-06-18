import random
import numpy as np
import pandas as pd
import os

# Cargar datos
df = pd.read_csv('algoritmos_evolutivos-s8_lab/notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# Crear cromosoma (permutación aleatoria de los 39 alumnos)
def crear_cromosoma():
    return random.sample(range(39), 39)

# Decodificar cromosoma a asignaciones de examen
def decodificar_cromosoma(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': []}
    for i in range(39):
        grupo = i // 13
        if grupo == 0:
            asignaciones['A'].append(cromosoma[i])
        elif grupo == 1:
            asignaciones['B'].append(cromosoma[i])
        else:
            asignaciones['C'].append(cromosoma[i])
    return asignaciones

# Calcular fitness del cromosoma
def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    promedios = []
    varianzas = []
    diversidad_bonus = 0

    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]

        promedio = np.mean(notas_examen)
        varianza = np.var(notas_examen)
        promedios.append(promedio)
        varianzas.append(varianza)

        rango = max(notas_examen) - min(notas_examen)
        if rango > 5:
            diversidad_bonus += 0.1

    desv_promedios = np.std(promedios)
    promedio_varianzas = np.mean(varianzas)

    fitness = -desv_promedios - 0.1 * promedio_varianzas + diversidad_bonus
    return fitness

# Mutación: intercambio de dos genes
def mutacion(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    i, j = random.sample(range(39), 2)
    cromosoma_mutado[i], cromosoma_mutado[j] = cromosoma_mutado[j], cromosoma_mutado[i]
    return cromosoma_mutado

# Algoritmo genético
def algoritmo_genetico(generaciones=100, tam_poblacion=50):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    mejores_fitness = []  # Lista para almacenar el mejor fitness de cada generación

    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        mejor_fitness = fitness_scores[0][1]
        mejores_fitness.append(mejor_fitness)  # Guardar mejor fitness de esta generación

        nueva_poblacion = []
        elite = int(tam_poblacion * 0.2)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])

        while len(nueva_poblacion) < tam_poblacion:
            padre = random.choice(poblacion[:tam_poblacion//2])
            hijo = mutacion(padre)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        if gen % 20 == 0:
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness:.4f}")

    mejor_cromosoma = fitness_scores[0][0]
    return mejor_cromosoma, mejores_fitness  # Devuelve cromosoma y lista de fitness

# Ejecución
print("REPRESENTACIÓN PERMUTACIONAL MODIFICADA")
print("Problema: Distribuir 39 alumnos en 3 exámenes (A, B, C) de forma equitativa")
print("Cromosoma: Permutación de 39 elementos (índices de alumnos)\n")

mejor_solucion, lista_fitness = algoritmo_genetico()

asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución final:")
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    print(f"Examen {examen}: {len(indices)} alumnos, promedio = {promedio:.2f}")
    print(f"  Alumnos: {[alumnos[i] for i in indices[:5]]}... (mostrando primeros 5)")

print("\nVerificación de equilibrio:")
promedios = []
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")

# Crear carpeta de salida si no existe
output_dir = 'algoritmos_evolutivos-s8_lab'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar resultados
notas_dict = {}
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_dict[examen] = [notas[i] for i in indices]

np.savez(os.path.join(output_dir, 'resultado_permutacional_modificada.npz'),
         fitness=lista_fitness,  # Guardar toda la evolución del fitness
         asignaciones=asignaciones_finales,
         notas=notas_dict)  # Diccionario compatible con visualización
