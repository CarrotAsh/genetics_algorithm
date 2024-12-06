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

def parent_by_tournament(population, fitness, *args, **kwargs):
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
    selected_parents = set()#los parents que ya han aparecido para que no se repitan
    while len(parents) < number_parents: #hacemos el bucle hasta alcanzar el numero de padres

        parent = parent_by_tournament(population, fitness, *args, **kwargs)
        parent_tuple = tuple(parent) #lo convertimos en tupla

        if parent_tuple not in selected_parents:
          parents.append(parent) #agregamos el padre que gana la seleccion por torneo
          selected_parents.add(parent_tuple)

    return parents #devolvemos la lista que sera la nueva generacion

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
    return generation > max_gen

def genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                      selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Aplica un algoritmo genético a un problema de maximización
    population = None # Crea la población de individuos de tamaño pop_size
    fitness = None # Contiene la evaluación de la población
    best_fitness = [] # Guarda el mejor fitness de cada generación
    mean_fitness = [] # Guarda el fitness medio de cada generación
    generation = 0 # Contador de generaciones

    # 1 - Inicializa la población con la función generate_population
    # 2 - Evalúa la población con la función fitness_function
    # 3 - Mientras no se cumpla el criterio de parada stopping_criteria
    # 4 - Selección de padres con la función selection
    # 5 - Cruce de padres mediante la función crossover con probabilidad p_cross
    # 6 - Mutación de los descendientes con la función mutation con probabilidad p_mut
    # 7 - Evaluación de los descendientes
    # 8 - Generación de la nueva población con la función environmental_selection

    return population, fitness, generation, best_fitness, mean_fitness

### Coloca aquí tus funciones propuestas para la generación de población inicial ###

### Coloca aquí tus funciones de fitness propuestas ###

### Coloca aquí tus funciones de selección propuestas ###

### Coloca aquí tus funciones de cruce propuestas ###

### Coloca aquí tus funciones de mutación propuestas ###

### Coloca aquí tus funciones de reemplazo propuestas ###

### Coloca aquí tus funciones de parada propuestas ###


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