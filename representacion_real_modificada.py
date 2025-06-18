import random
import numpy as np
import pandas as pd
import os

# Cargar datos
df = pd.read_csv('algoritmos_evolutivos-s8_lab/notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# Crear cromosoma: para cada alumno 3 genes (A, B, C) normalizados
def crear_cromosoma():
    cromosoma = []
    for i in range(39):
        pesos = [random.random() for _ in range(3)]
        suma = sum(pesos)
        pesos_norm = [p/suma for p in pesos]
        cromosoma.extend(pesos_norm)
    return cromosoma

# Decodificación: asignar cada alumno al examen con mayor probabilidad
def decodificar_cromosoma(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': []}
    examenes = ['A', 'B', 'C']
    alumnos_disponibles = list(range(39))
    contadores = {'A': 0, 'B': 0, 'C': 0}
    
    while alumnos_disponibles:
        mejor_alumno = None
        mejor_examen = None
        mejor_valor = -1
        for alumno in alumnos_disponibles:
            idx = alumno * 3
            for i, examen in enumerate(examenes):
                if contadores[examen] < 13:
                    valor = cromosoma[idx + i]
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_alumno = alumno
                        mejor_examen = examen
        if mejor_alumno is not None:
            asignaciones[mejor_examen].append(mejor_alumno)
            contadores[mejor_examen] += 1
            alumnos_disponibles.remove(mejor_alumno)
    return asignaciones

# Calcular fitness
def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    promedios = {}
    varianzas = {}
    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
        varianzas[examen] = np.var(notas_examen)
    desv_promedios = np.std(list(promedios.values()))
    promedio_varianzas = np.mean(list(varianzas.values()))
    fitness = -desv_promedios - 0.1 * promedio_varianzas
    return fitness

# Cruce
def cruce(padre1, padre2):
    hijo = []
    for i in range(39):
        idx = i * 3
        if random.random() < 0.5:
            genes = padre1[idx:idx+3]
        else:
            genes = padre2[idx:idx+3]
        genes = [g + random.gauss(0, 0.1) for g in genes]
        genes = [max(0, g) for g in genes]
        suma = sum(genes)
        if suma > 0:
            genes = [g/suma for g in genes]
        else:
            genes = [1/3, 1/3, 1/3]
        hijo.extend(genes)
    return hijo

# Mutación gaussiana
def mutacion_gaussiana(cromosoma, sigma=0.1):
    cromosoma_mutado = cromosoma.copy()
    for i in range(39):
        idx = i * 3
        genes = [cromosoma_mutado[idx + j] + random.gauss(0, sigma) for j in range(3)]
        genes = [max(0, g) for g in genes]
        suma = sum(genes)
        if suma > 0:
            genes = [g/suma for g in genes]
        else:
            genes = [1/3, 1/3, 1/3]
        cromosoma_mutado[idx:idx+3] = genes
    return cromosoma_mutado

# Algoritmo genético con historial de fitness
def algoritmo_genetico(generaciones=150, tam_poblacion=100, sigma=0.1):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    mejor_global_fitness = float('-inf')
    mejor_global_cromosoma = None
    historial_fitness = []  # ⬅️ Guardamos el fitness por generación

    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        if fitness_scores[0][1] > mejor_global_fitness:
            mejor_global_fitness = fitness_scores[0][1]
            mejor_global_cromosoma = fitness_scores[0][0].copy()

        historial_fitness.append(fitness_scores[0][1])  # ⬅️ Guardamos mejor fitness de esta generación
        
        nueva_poblacion = []
        elite = int(tam_poblacion * 0.1)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])
        
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = random.choice(poblacion[:tam_poblacion//4])
            padre2 = random.choice(poblacion[:tam_poblacion//4])
            hijo = cruce(padre1, padre2)
            hijo = mutacion_gaussiana(hijo, sigma=sigma)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion

        if gen % 30 == 0:
            print(f"Generación {gen}: Mejor fitness = {fitness_scores[0][1]:.4f}")

    return mejor_global_cromosoma, historial_fitness

# Ejecución
print("REPRESENTACIÓN REAL CON MUTACIÓN GAUSSIANA")
sigma_valor = 0.5
mejor_solucion, historial_fitness = algoritmo_genetico(sigma=sigma_valor)
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución optimizada:")
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    varianza = np.var(notas_examen)
    print(f"Examen {examen}: {len(indices)} alumnos")
    print(f"  Promedio: {promedio:.2f}, Varianza: {varianza:.2f}")
    print(f"  Rango de notas: [{min(notas_examen):.0f} - {max(notas_examen):.0f}]")

print("\nAnálisis de equilibrio:")
promedios = []
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))

print(f"Promedios por examen: A={promedios[0]:.2f}, B={promedios[1]:.2f}, C={promedios[2]:.2f}")
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")
print(f"Diferencia máxima entre promedios: {max(promedios) - min(promedios):.2f}")

# Crear carpeta de salida si no existe
output_dir = 'algoritmos_evolutivos-s8_lab'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 🔍 Guardamos notas como diccionario {A:[], B:[], C:[]}
notas_dict = {}
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_dict[examen] = [notas[i] for i in indices]

# Guardar resultado con historial de fitness
np.savez(os.path.join(output_dir, 'resultado_real_modificada.npz'),
         fitness=historial_fitness,   # ⬅️ Ahora guarda historial completo
         asignaciones=asignaciones_finales,
         notas=notas_dict)
