import math
import numpy as np
import random

# Ejemplo de dataset de entrada para el problema de asignación de horarios
dataset = {"n_courses" : 3,
           "n_days" : 3,
           "n_hours_day" : 3,
           "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

def generate_random_array_int(alphabet, length):
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]
    # Genera un array de enteros aleatorios de tamaño length
    # usando el alfabeto dado

def generate_initial_population_timetabling(pop_size, *args, **kwargs):
    dataset = kwargs['dataset'] # Dataset con la misma estructura que el ejemplo

    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    alphabet = list(range(n_days * n_hours_day))

    individual_length = sum(hours for _, hours in courses)

    population = [generate_random_array_int(alphabet, individual_length) for _ in range(pop_size)]
    # Obtener el alfabeto y la longitud a partir del dataset
    # Genera una población inicial de tamaño pop_size
    return population

################################# NO TOCAR #################################
#                                                                          #
def print_timetabling_solution(solution, dataset):
    # Imprime una solución de timetabling
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Crea una matriz de n_days x n_hours_day
    timetable = [[[] for _ in range(n_hours_day)] for _ in range(n_days)]

    # Llena la matriz con las asignaturas
    i = 0
    max_len = 6 # Longitud del título Día XX
    for course in courses:
        for _ in range(course[1]):
            day = solution[i] // n_hours_day
            hour = solution[i] % n_hours_day
            timetable[day][hour].append(course[0])
            i += 1
            # Calcula la longitud máxima del nombre de las asignaturas
            # en una misma franja horaria
            max_len = max(max_len, len('/'.join(timetable[day][hour])))

    # Imprime la matriz con formato de tabla markdown
    print('|         |', end='')
    for i in range(n_days):
        print(f' Día {i+1:<2}{" "*(max_len-6)} |', end='')
    print()
    print('|---------|', end='')
    for i in range(n_days):
        print(f'-{"-"*max_len}-|', end='')
    print()
    for j in range(n_hours_day):
        print(f'| Hora {j+1:<2} |', end='')
        for i in range(n_days):
            s = '/'.join(timetable[i][j])
            print(f' {s}{" "*(max_len-len(s))}', end=' |')
        print()
#                                                                          #
################################# NO TOCAR #################################

# Ejemplo de uso de la función generar individuo con el dataset de ejemplo
candidate = generate_random_array_int(list(range(9)), 6)
print_timetabling_solution(candidate, dataset)

def create_timetable(solution, dataset): #Crea una matriz con el num de asignaturas en cada franja horaria
    n_hours_day = dataset['n_hours_day']
    n_days = dataset['n_days']

    timetable = np.empty((n_hours_day, n_days) , dtype=object)
    for i in range(n_days):
        for j in range(n_hours_day):
            timetable[i][j] = []

    i = 0
    for course in dataset['courses']:
        n_hours_subject = course[1]
        for _ in range(n_hours_subject):
            day = solution[i] // n_hours_day
            hour = solution[i] % n_hours_day
            timetable[hour][day].append(course[0])
            i += 1

    return timetable

def hoursPerSubject(data):
    hours =[]
    courses = data['courses']



    return hours

def calculate_c1(solution, *args, **kwargs):
    dataset = kwargs['dataset']
    conflicts = 0
    timetable = create_timetable(solution, dataset)

    for row in timetable:
        for value in row:
            if len(value) > 1:
                conflicts += len(value) - 1

    return conflicts

def calculate_c2(solution, *args, **kwargs):
    dataset = kwargs['dataset']
    timetable = create_timetable(solution, dataset)
    hours = 0

    if dataset['n_hours_day'] <= 2:
        return 0

    for course in dataset['courses']:
        subject = course[0]

        for j in range(dataset['n_hours_day']):
            n_hours_per_subject_day = 0
            for i in range(dataset['n_days']):
                if subject in timetable[i][j]:
                    n_hours_per_subject_day += 1
                    if n_hours_per_subject_day > 2:
                        hours += 1

    # Calcula la cantidad de horas por encima de 2 que se imparten
    # de una misma asignatura en un mismo día
    return hours

def calculate_p1(solution, *args, **kwargs):
    dataset = kwargs['dataset']
    timetable = create_timetable(solution, dataset)

    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    gaps = 0

    for day in range(n_days):
        day_schedule = [len(timetable[hour][day]) > 0 for hour in range(n_hours_day)]

        if any(day_schedule):
            first_occupied = day_schedule.index(True)
            last_occupied = len(day_schedule) - 1 - day_schedule[::-1].index(True)
            gaps += sum(1 for i in range(first_occupied, last_occupied + 1) if not day_schedule[i])

    # Calcula el número de huecos vacíos entre asignaturas

    return gaps

def calculate_p2(solution, *args, **kwargs): #DA FALLOS
    dataset = kwargs['dataset']
    timetable = create_timetable(solution, dataset)
    days_used = 0

    for j in range(timetable.shape[1]):
        for i in range(timetable.shape[0]):
            if len(timetable[i][j]) >=1:
                days_used += 1
                break

    # Calcula el número de días utilizados en los horarios
    return days_used

def calculate_p3(solution, *args, **kwargs):
    dataset = kwargs['dataset']
    timetable = create_timetable(solution, dataset)

    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    non_consecutive_count = 0

    # Recorrer cada asignatura
    for course, _ in dataset['courses']:
        # Revisar cada día
        for day in range(n_days):
            day_schedule = [course in timetable[hour][day] for hour in range(n_hours_day)]

            if any(day_schedule):  # Si la asignatura está presente ese día
                # Encontrar el rango ocupado
                first_occupied = day_schedule.index(True)
                last_occupied = len(day_schedule) - 1 - day_schedule[::-1].index(True)

                # Verificar si hay huecos dentro del rango ocupado
                if any(not day_schedule[i] for i in range(first_occupied, last_occupied + 1)):
                    non_consecutive_count += 1

    # Calcula el número de asignaturas con horas NO consecutivas en un mismo día
    return non_consecutive_count

def fitness_timetabling(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    c1 = calculate_c1(solution, **kwargs)
    c2 = calculate_c2(solution, **kwargs)

    if c1 > 0 or c2 > 0:
        return 0

    p1 = calculate_p1(solution, **kwargs)
    p2 = calculate_p2(solution, **kwargs)
    p3 = calculate_p3(solution, **kwargs)

    fitness_value = 1 / (1 + p1 + p2 + p3)  # Calculamos la función fitness según la primera aproximación

    return fitness_value

conflicts = calculate_c1(candidate, dataset=dataset)
hours = calculate_c2(candidate, dataset=dataset)
gaps = calculate_p1(candidate, dataset=dataset)
days_used = calculate_p2(candidate, dataset=dataset)
non_consecutive_subjects = calculate_p3(candidate, dataset=dataset)
fitness_value = fitness_timetabling(candidate, dataset=dataset) # Devuelve la fitness del candidato de ejemplo

print()
print("Conflicts", conflicts)
print("More than 2 consecutive hours", hours)
print("Gaps between subjects", gaps)
print("Days used", days_used)
print("Non consecutive subjects", non_consecutive_subjects)
print("Fitness value: ", fitness_value)
print()

# Pistas:
# - Una función que devuelva la tabla de horarios de una solución
# - Una función que devuelva la cantidad de horas por día de cada asignatura
# - A través de args y kwargs se pueden pasar argumentos adicionales que vayamos a necesitar

def parent_by_tournament(population, fitness, *args, **kwargs): #Función auxiliar que genera un padre a partir de una selección por torneo
    tournament_size = kwargs['tournament_size']  # Tamaño del torneo

    # Seleccionamos aleatoriamente 'tournament_size' individuos de la población
    competitors = random.sample(population, tournament_size)

    # Evaluamos el fitness de cada competidor en el torneo
    fitness_values = [fitness(individual, *args, **kwargs) for individual in competitors]

    # Seleccionamos al individuo con el mejor fitness (máximo fitness en este caso)
    parent = competitors[fitness_values.index(max(fitness_values))]

    return parent

def tournament_selection(population, fitness, number_parents, *args, **kwargs):
    # Selecciona number_parents individuos de la población mediante selección por torneo
    parents = []
    #selected_parents = set()#los parents que ya han aparecido para que no se repitan
    while len(parents) < number_parents: #hacemos el bucle hasta alcanzar el numero de padres

        parent = parent_by_tournament(population, fitness, *args, **kwargs)
        #parent_tuple = tuple(parent) #lo convertimos en tupla

        #if parent_tuple not in selected_parents:
        parents.append(parent) #agregamos el padre que gana la seleccion por torneo
          #selected_parents.add(parent_tuple)

    return parents #devolvemos la lista que sera la nueva generacion

# Pista:
# - Crear una función auxiliar que genere un padre a partir de una selección por torneo
# - Recuerda usar la misma librería de números aleatorios que en el resto del código

#Comprobamos que funciona, pop size del ejemplo es 4 BORRAR LUEGO
initial_population = generate_initial_population_timetabling(4,dataset=dataset)
for i, candidate in enumerate(initial_population):
    print(f"Timetable: {i + 1}:")
    print_timetabling_solution(candidate, dataset=dataset)
    print()

parents = tournament_selection(initial_population, fitness_timetabling, 2, tournament_size=2, dataset=dataset)
print("Parents:")
for i, parent in enumerate(parents):
    print(f"Parent {i + 1}:")
    print_timetabling_solution(parent, dataset = dataset)
    print()

# Pista:
# - Crear una función auxiliar que genere un padre a partir de una selección por torneo
# - Recuerda usar la misma librería de números aleatorios que en el resto del código

def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    # Realiza el cruce de dos padres con una probabilidad p_cross

    child1 = parent1.copy()
    child2 = parent2.copy()

    if random.random() >= p_cross:
        return child1, child2

    cross_point = random.randint(1, len(child1) - 1)

    for i in range(cross_point, len(child1)):
        aux = child1[i]
        child1[i] = child2[i]
        child2[i] = aux

    return child1, child2

def uniform_mutation(chromosome, p_mut, *args, **kwargs):
    dataset = kwargs['dataset'] # Dataset con la misma estructura que el ejemplo
    # Realiza la mutación gen a gen con una probabilidad p_mut
    # Obtener el alfabeto del dataset para aplicar la mutación

    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']

    alphabet = list(range(n_days * n_hours_day))

    for i in range(len(chromosome)):
        if random.random() >= p_mut:
            continue

        chromosome[i] = alphabet[random.randint(0, len(alphabet) - 1)]

    return chromosome

def generational_replacement(population, fitness, offspring, fitness_offspring, *args, **kwargs):
    # Realiza la sustitución generacional de la población
    # Debe devolver tanto la nueva población como el fitness de la misma

    pop_fit = list(zip(population, fitness))
    pop_fit.sort(key=lambda x: x[1])

    off_fit = list(zip(population, fitness_offspring))

    for i in range(len(offspring)):
        pop_fit[i] = off_fit[i]

    new_population, new_fitness = zip(*pop_fit)

    return new_population, new_fitness

def generation_stop(generation, fitness, *args, **kwargs):
    max_gen=kwargs['max_gen']
    # Comprueba si se cumple el criterio de parada (máximo número de generaciones)
    return generation >= max_gen

def genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                      selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Aplica un algoritmo genético a un problema de maximización
    population = None # Crea la población de individuos de tamaño pop_size
    fitness_values = None # Contiene la evaluación de la población
    best_fitness = [] # Guarda el mejor fitness de cada generación
    mean_fitness = [] # Guarda el fitness medio de cada generación
    generation = 0 # Contador de generaciones

    # 1 - Inicializa la población con la función generate_population
    population = generate_population(pop_size, *args, **kwargs)

    # 2 - Evalúa la población con la función fitness_function
    fitness_values = [fitness_function(x, *args, **kwargs) for x in population]
    best_fitness.append(np.max(fitness_values))
    mean_fitness.append(np.mean(fitness_values))

    # 3 - Mientras no se cumpla el criterio de parada stopping_criteria
    while not stopping_criteria(generation, fitness_values, *args, **kwargs):
        # 4 - Selección de padres con la función selection
        parents = selection(population, fitness_function, offspring_size if (offspring_size%2==0) else offspring_size+1, *args, **kwargs)

        # 5 - Cruce de padres mediante la función crossover con probabilidad p_cross
        offspring = []
        for k in range(math.ceil(offspring_size / 2)):
            parent1 = parents[2 * k]
            parent2 = parents[2 * k + 1]
            child1, child2 = crossover(parent1, parent2, p_cross, *args, **kwargs)

        # 6 - Mutación de los descendientes con la función mutation con probabilidad p_mut
            child1 = mutation(child1, p_mut, *args, **kwargs)
            offspring.append(child1)
            if 2 * k + 1 < offspring_size:
                child2 = mutation(child2, p_mut, *args, **kwargs)
                offspring.append(child2)

        # 7 - Evaluación de los descendientes
            fitness_offspring = [fitness_function(x, *args, **kwargs) for x in offspring]
            best_fitness.append(np.max(fitness_offspring))
            mean_fitness.append(np.mean(fitness_offspring))

        # 8 - Generación de la nueva población con la función environmental_selection
            population, fitness_values = environmental_selection(population, fitness_values, offspring, fitness_offspring, *args, **kwargs)

        generation += 1

    return population, fitness_values, generation, best_fitness, mean_fitness

'''
En nuestra función para la generación de la población en la aproximación final hemos agregado una restricción, 
para evitar conflictos de horario entre asignaturas. Cuando tomamos un valor aleatorio del alfabeto este es eliminado
de los valores posibles del mismo (para ese individuo) al contrario que en la función de la aproximación inicial,
en la cual se permite repetir valores. 
'''

### Coloca aquí tus funciones propuestas para la generación de población inicial ###
def generate_initial_population_final(pop_size, *args, **kwargs):
    dataset = kwargs['dataset']
    courses = dataset['courses']
    population = []

    individual_length = sum(hours for _, hours in courses)

    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']

    alphabet = list(range(n_days * n_hours_day))
    individual_length = sum(hours for _, hours in courses)

    for i in range(pop_size):
        alphabet_copy = alphabet.copy()
        individual = np.zeros(individual_length, dtype=int)

        for j in range(individual_length):
            individual[j] = alphabet_copy.pop(random.randint(0, individual_length - 1))

        population.append(individual)

    return population

'''
En vez de dar un cero como fitness del individuo, se le da unos pesos a las restricciones para que se pueda 
diferenciar entre un individuo decente que no cumple las restricciones y uno malo que tampoco las pase, 
de esta forma no se descartan individuos que no las cumplen por encima de otros con preferencias buenas.
'''
### Coloca aquí tus funciones de fitness propuestas ###
def fitness_timetabling_final(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    p1 = calculate_p1(solution, **kwargs)
    p2 = calculate_p2(solution, **kwargs)
    p3 = calculate_p3(solution, **kwargs)

    fitness_value = 1 / (1 + p1 + p2 + p3)

    c1_weighted = calculate_c1(solution, **kwargs) * 4
    c2_weighted = calculate_c2(solution, **kwargs) * 2

    fitness_value /=  1 + c1_weighted + c2_weighted

    return fitness_value

### Coloca aquí tus funciones de selección propuestas ###

### Coloca aquí tus funciones de cruce propuestas ###

'''
Para cruzar a dos individuos elegimos un indice aleatorio y cambiamos los valores que esten en ese indice entre
un array y otro.
Esto sirve para cambiar una clase de la franja horaria en la que esta en este individuo
a la franja horaria en la que este en el otro individuo. 
Si hay conflicto porque ya habia una clase en esa franja horaria la cambiamos a la otra.

EJ:
    parent1 = [1,4,3] -> [2,4,3]
               ^
               |
               v
    parent2 = [2,3,1] -> [1,3,2]
    
    Cambiamos el 1 y el 2, y si el valor entrante ya se encuentra en el array lo sustituimos por el valor saliente
'''
def change_values_cross_final(parent1, parent2, p_cross, *args, **kwargs):
    child1 = parent1.copy()
    child2 = parent2.copy()

    if random.random() >= p_cross:
        return child1, child2

    cross_index = child1[random.randint(0, len(child1))]
    value1 = child1[cross_index]
    value2 = child2[cross_index]

    # Si la clase esta en la misma franja horaria el cruce no tiene efecto (otra forma de hacerlo seria probando clases
    # hasta que haya una diferencia)
    if value1 == value2:
        return child1, child2

    child1[cross_index], child2[cross_index] = child2[cross_index], child1[cross_index]

    child1[child1 == value2] = value1
    child2[child2 == value1] = value2

    return child1, child2

### Coloca aquí tus funciones de mutación propuestas ###

'''
Introduce solo valores nuevos al mutar para evitar conflictos.
'''
def only_new_values_mutation_final(chromosome, p_mut, *args, **kwargs):
    n_days = dataset['n_days']
    n_hours_days = dataset['n_hours_day']

    alphabet = list(range(n_days * n_hours_days))

    not_in_chromosome = [x for x in alphabet if x not in chromosome]

    for i in range(len(chromosome)):
        if random.random() >= p_mut:
            continue

        chromosome[i] = not_in_chromosome.pop(random.randint(0, len(not_in_chromosome) - 1))

        if len(not_in_chromosome) == 0:
            not_in_chromosome = [x for x in alphabet if x not in chromosome]

    return chromosome

### Coloca aquí tus funciones de reemplazo propuestas ###

### Coloca aquí tus funciones de parada propuestas ###

'''
Para mejorar la condicion de parada añadimos la condicion de que pare cuando la mejor fitness 
no haya mejorado en 4 generaciones. De tal forma evitamos seguir iterando sin conseguir mejores individuos
'''

def generation_stop_final(generation, fitness, *args, **kwargs):
    max_gen=kwargs['max_gen']
    max_fit_gen = max(fitness)
    max_fit = 0
    counter = -1

    while True:
        if max_fit_gen > max_fit:
            counter += 1
            if counter >= 3:
                return False
        yield generation >= max_gen
        max_fit_gen = max(fitness)

'''
################################# NO TOCAR #################################
#                                                                          #
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        return *res, end - start
    return wrapper
#                                                                          #
################################# NO TOCAR #################################

# Este codigo temporiza la ejecución de una función cualquiera

################################# NO TOCAR #################################
#                                                                          #
@timer
def run_ga(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
           selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Además del retorno de la función, se devuelve el tiempo de ejecución en segundos
    return genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                             selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs)
#                                                                          #
################################# NO TOCAR #################################

# Se deben probar los 6 datasets
dataset1 = {"n_courses" : 3,
            "n_days" : 3,
            "n_hours_day" : 3,
            "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

dataset2 = {"n_courses" : 4,
            "n_days" : 3,
            "n_hours_day" : 4,
            "courses" : [("IA", 1), ("ALG", 2), ("BD", 3), ("POO", 2)]}

dataset3 = {"n_courses" : 4,
            "n_days" : 4,
            "n_hours_day" : 4,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4)]}

dataset4 = {"n_courses" : 5,
            "n_days" : 4,
            "n_hours_day" : 6,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4)]}

dataset5 = {"n_courses" : 7,
            "n_days" : 4,
            "n_hours_day" : 8,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4), ("FP", 4), ("TP", 2)]}

dataset6 = {"n_courses" : 11,
            "n_days" : 5,
            "n_hours_day" : 12,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4), ("FP", 4), ("TP", 2), ("FC", 4), ("TSO", 2), ("AM", 4), ("LMD", 4)]}

import numpy as np
import random

def set_seed(seed):
    # Se debe fijar la semilla usada para generar números aleatorios
    # Con la librería random
    random.seed(seed)
    # Con la librería numpy
    np.random.seed(seed)

################################# NO TOCAR #################################
#                                                                          #
def best_solution(population, fitness):
    # Devuelve la mejor solución de la población
    return population[fitness.index(max(fitness))]

import matplotlib.pyplot as plt
def plot_fitness_evolution(best_fitness, mean_fitness):
    plt.plot(best_fitness, label='Best fitness')
    plt.plot(mean_fitness, label='Mean fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
#                                                                          #
################################# NO TOCAR #################################

from statistics import mean, median, stdev

def launch_experiment(seeds, dataset, generate_population, pop_size, fitness_function, c1, c2, p1, p2, p3, stopping_criteria,
                      offspring_size, selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    best_individuals = []
    best_inds_c1 = []
    best_inds_c2 = []
    best_inds_p1 = []
    best_inds_p2 = []
    best_inds_p3 = []
    best_inds_fitness = []
    best_fitnesses = []
    mean_fitnesses = []
    last_generations = []
    execution_times = []
    # Ejecutamos el algoritmo con cada semilla
    for seed in seeds:
        print(f"Running Genetic Algorithm with seed {seed}")
        set_seed(seed)
        population, fitness, generation, best_fitness, mean_fitness, execution_time = run_ga(generate_population, pop_size, fitness_function,stopping_criteria,
                                                                                             offspring_size, selection, crossover, p_cross, mutation, p_mut,
                                                                                             environmental_selection, dataset=dataset, *args, **kwargs)
        best_individual = best_solution(population, fitness)
        best_ind_c1 = c1(best_individual, dataset=dataset)
        best_ind_c2 = c2(best_individual, dataset=dataset)
        best_ind_p1 = p1(best_individual, dataset=dataset)
        best_ind_p2 = p2(best_individual, dataset=dataset)
        best_ind_p3 = p3(best_individual, dataset=dataset)
        best_ind_fitness = fitness_function(best_individual, dataset=dataset)
        best_individuals.append(best_individual)
        best_inds_c1.append(best_ind_c1)
        best_inds_c2.append(best_ind_c2)
        best_inds_p1.append(best_ind_p1)
        best_inds_p2.append(best_ind_p2)
        best_inds_p3.append(best_ind_p3)
        best_inds_fitness.append(best_ind_fitness)
        best_fitnesses.append(best_fitness)
        mean_fitnesses.append(mean_fitness)
        last_generations.append(generation)
        execution_times.append(execution_time)
    # Imprimimos la media y desviación típica de los resultados obtenidos
    print("Mean Best Fitness: " + str(mean(best_inds_fitness)) + " " + u"\u00B1" + " " + str(stdev(best_inds_fitness)))
    print("Mean C1: " + str(mean(best_inds_c1)) + " " + u"\u00B1" + " " + str(stdev(best_inds_c1)))
    print("Mean C2: " + str(mean(best_inds_c2)) + " " + u"\u00B1" + " " + str(stdev(best_inds_c2)))
    print("Mean P1: " + str(mean(best_inds_p1)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p1)))
    print("Mean P2: " + str(mean(best_inds_p2)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p2)))
    print("Mean P3: " + str(mean(best_inds_p3)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p3)))
    print("Mean Execution Time: " + str(mean(execution_times)) + " " + u"\u00B1" + " " + str(stdev(execution_times)))
    print("Mean Number of Generations: " + str(mean(last_generations)) + " " + u"\u00B1" + " " + str(stdev(last_generations)))
    # Mostramos la evolución de la fitness para la mejor ejecución
    print("Best execution fitness evolution:")
    best_execution = best_inds_fitness.index(max(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[best_execution], mean_fitnesses[best_execution])
    # Mostramos la evolución de la fitness para la ejecución mediana
    print("Median execution fitness evolution:")
    median_execution = best_inds_fitness.index(median(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[median_execution], mean_fitnesses[median_execution])
    # Mostramos la evolución de la fitness para la peor ejecución
    print("Worst execution fitness evolution:")
    worst_execution = best_inds_fitness.index(min(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[worst_execution], mean_fitnesses[worst_execution])

    return best_individuals, best_inds_fitness, best_fitnesses, mean_fitnesses, last_generations, execution_times

# Crear un conjunto de 31 semillas para los experimentos
seeds = [1234567890 + i*23 for i in range(31)] # Semillas de ejemplo, cambiar por las semillas que se quieran
launch_experiment(seeds, dataset1, generate_initial_population_timetabling, 50, fitness_timetabling, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop, 50, tournament_selection, one_point_crossover, 0.8,
                  uniform_mutation, 0.1, generational_replacement, max_gen=50, tournament_size=2)
# Recuerda también mostrar el horario de la mejor solución obtenida en los casos peor, mejor y mediano

### Coloca aquí tus experimentos ###

### Coloca aquí tus experimentos ###
'''