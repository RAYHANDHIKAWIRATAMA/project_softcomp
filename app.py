from flask import Flask, render_template, jsonify, request
import random
import numpy as np
import pandas as pd


app = Flask(__name__)

# ----------------------------------------------
# Data Masalah Knapsack
# ----------------------------------------------
items = {
    'A': {'weight': 7, 'value': 5},
    'B': {'weight': 2, 'value': 4},
    'C': {'weight': 1, 'value': 7},
    'D': {'weight': 9, 'value': 2}
}
capacity = 15
item_list = list(items.keys())
n_items = len(item_list)

# ----------------------------------------------
# Fungsi bantu Algoritma Genetika
# ----------------------------------------------
def decode(chromosome):
    """Kembalikan list item, total berat, total nilai"""
    total_weight = 0
    total_value = 0
    chosen_items = []
    for gene, name in zip(chromosome, item_list):
        if gene == 1:
            total_weight += items[name]['weight']
            total_value += items[name]['value']
            chosen_items.append(name)
    return chosen_items, total_weight, total_value

def fitness(chromosome):
    """Fungsi fitness dengan penalti berat"""
    _, total_weight, total_value = decode(chromosome)
    if total_weight <= capacity:
        return total_value
    else:
        return 0

def roulette_selection(population, fitnesses):
    """Seleksi roulette wheel"""
    total_fit = sum(fitnesses)
    if total_fit == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fit)
    current = 0
    for chrom, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return chrom
    return population[-1]

def crossover(p1, p2):
    """Single-point crossover"""
    if len(p1) != len(p2):
        raise ValueError("Parent length mismatch")
    point = random.randint(1, len(p1) - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(chromosome, mutation_rate=0.1):
    """Flip bit dengan probabilitas mutation_rate"""
    return [1 - g if random.random() < mutation_rate else g for g in chromosome]

# ----------------------------------------------
# Algoritma Genetika Utama
# ----------------------------------------------
def genetic_algorithm(pop_size=10, generations=10, crossover_rate=0.8, mutation_rate=0.1, elitism=True):
    population = [[random.randint(0, 1) for _ in range(n_items)] for _ in range(pop_size)]
    
    generation_data = []
    
    for gen in range(generations):
        fitnesses = [fitness(ch) for ch in population]
        best_index = fitnesses.index(max(fitnesses))
        best_chrom = population[best_index]
        best_fit = fitnesses[best_index]
        best_items, w, v = decode(best_chrom)
        
        generation_data.append({
            'generation': gen + 1,
            'chromosome': best_chrom,
            'items': best_items,
            'weight': w,
            'value': v,
            'fitness': best_fit
        })
        
        new_population = []
        if elitism:
            new_population.append(best_chrom)
        
        while len(new_population) < pop_size:
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
    
    fitnesses = [fitness(ch) for ch in population]
    best_index = fitnesses.index(max(fitnesses))
    best_chrom = population[best_index]
    best_items, w, v = decode(best_chrom)
    
    return {
        'generations': generation_data,
        'final': {
            'chromosome': best_chrom,
            'items': best_items,
            'weight': w,
            'value': v,
            'fitness': fitness(best_chrom)
        }
    }

# ----------------------------------------------
# Algoritma Genetika TSP (untuk Minggu 4)
# ----------------------------------------------

def route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i], route[(i+1)%len(route)]] for i in range(len(route)))

def create_individual(n):
    ind = list(range(n))
    random.shuffle(ind)
    return ind

def initial_population(size, n):
    return [create_individual(n) for _ in range(size)]

def tournament_selection(pop, dist_matrix, k=5):
    selected = random.sample(pop, k)
    return min(selected, key=lambda ind: route_distance(ind, dist_matrix))

def ordered_crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [-1] * len(p1)
    child[a:b+1] = p1[a:b+1]

    p2_idx = 0
    for i in range(len(p1)):
        if child[i] == -1:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
    return child

def swap_mutation(ind):
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]

def ga_tsp(dist_matrix, cities, POP_SIZE=200, GENERATIONS=150, PC=0.8, PM=0.2, ELITE=1):
    n = len(cities)
    pop = initial_population(POP_SIZE, n)
    
    best = min(pop, key=lambda ind: route_distance(ind, dist_matrix))
    best_dist = route_distance(best, dist_matrix)

    history = []

    for g in range(GENERATIONS):
        pop = sorted(pop, key=lambda ind: route_distance(ind, dist_matrix))

        if route_distance(pop[0], dist_matrix) < best_dist:
            best = pop[0]
            best_dist = route_distance(best, dist_matrix)

        new_pop = pop[:ELITE]

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, dist_matrix)
            p2 = tournament_selection(pop, dist_matrix)

            child = ordered_crossover(p1, p2) if random.random() < PC else p1[:]

            if random.random() < PM:
                swap_mutation(child)

            new_pop.append(child)

        pop = new_pop
        history.append(best_dist)

    best_route = [cities[i] for i in best]

    return {
        "best_route": best_route,
        "distance": best_dist,
        "history": history
    }


# ----------------------------------------------
# Routes
# ----------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', title="Beranda")

@app.route('/minggu1')
def minggu1():
    return render_template('minggu1.html', title="Minggu 1 : Soft Computing")

@app.route('/minggu2')
def minggu2():
    return render_template('minggu2.html', title="Minggu 2 : Fuzzy Sugeno")

@app.route('/minggu3')
def minggu3():
    return render_template('minggu3.html', title="Minggu 3 : Algoritma Genetika")

@app.route('/minggu4')
def minggu4():
    return render_template('minggu4.html', title="Minggu 4")

@app.route('/run_ga', methods=['POST'])
def run_ga():
    data = request.json
    pop_size = int(data.get('pop_size', 8))
    generations = int(data.get('generations', 8))
    crossover_rate = float(data.get('crossover_rate', 0.8))
    mutation_rate = float(data.get('mutation_rate', 0.1))
    
    random.seed(42)
    result = genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate)
    
    return jsonify({
        'success': True,
        'data': result,
        'items_info': items,
        'capacity': capacity
    })

@app.route('/run_tsp', methods=['POST'])
def run_tsp():
    data = request.json

    city_names = data.get("cities")
    matrix = data.get("matrix")

    if city_names is None or matrix is None:
        return jsonify({'success': False, 'message': 'Data tabel tidak ditemukan'}), 400

    df = pd.DataFrame(matrix, columns=city_names, index=city_names)
    dist_matrix = df.values.astype(float)

    POP = int(data.get('pop', 200))
    GEN = int(data.get('gen', 150))
    PC = float(data.get('pc', 0.8))
    PM = float(data.get('pm', 0.2))

    random.seed(42)

    # Jalankan GA
    result = ga_tsp(dist_matrix, city_names, POP, GEN, PC, PM)

    return jsonify({
        'success': True,
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)
