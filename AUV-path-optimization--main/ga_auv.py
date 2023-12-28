import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

def calculate_fitness(waypoints, population, record_efficiency, best_path):
    current_record = float('inf')
    fitness = []

    for path in population:
        efficiency = calc_efficiency(waypoints, path)

        if efficiency < record_efficiency:
            record_efficiency = efficiency
            best_path = path

        if efficiency < current_record:
            current_record = efficiency

        # Fitness function
        fitness.append(1 / (pow(efficiency, 8) + 1))

    return fitness, record_efficiency, best_path

def normalize_fitness(fitness):
    total = sum(fitness)
    return [f / total for f in fitness]

def next_generation(population, fitness, mutation_rate=0.01):
    new_population = []

    for _ in range(len(population)):
        path_a = pick_one(population, fitness)
        path_b = pick_one(population, fitness)
        path = crossover(path_a, path_b)
        mutate(path, mutation_rate)
        new_population.append(path)

    return new_population

def pick_one(population, prob):
    index = 0
    r = random.random()

    while r > 0:
        r = r - prob[index]
        index += 1

    index -= 1
    return population[index][:]

def crossover(path_a, path_b):
    start = random.randint(0, len(path_a) - 1)
    end = random.randint(start + 1, len(path_a))
    new_path = path_a[start:end]

    for waypoint in path_b:
        if waypoint not in new_path:
            new_path.append(waypoint)

    return new_path

def mutate(path, mutation_rate):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            index_a = random.randint(0, len(path) - 1)
            index_b = (index_a + 1) % len(path)
            path[index_a], path[index_b] = path[index_b], path[index_a]

def calc_efficiency(waypoints, path):
    total_efficiency = 0

    for i in range(len(path) - 1):
        waypoint_a = path[i]
        waypoint_b = path[i + 1]
        # For simplicity, efficiency is calculated as Euclidean distance in this example.
        total_efficiency += euclidean_distance(waypoints[waypoint_a], waypoints[waypoint_b])

    return total_efficiency

def euclidean_distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

class AUVPathfindingVisualizer:
    def __init__(self, root, total_waypoints, population_size):
        self.root = root
        self.total_waypoints = total_waypoints
        self.population_size = population_size

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        self.start_button = ttk.Button(root, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack()

        self.waypoints = [[random.uniform(1, 10), random.uniform(1, 10)] for _ in range(total_waypoints)]
        self.population = [random.sample(range(total_waypoints), total_waypoints) for _ in range(population_size)]

        self.record_efficiency = float('inf')
        self.best_path = []

    def update_plot(self, path):
        self.ax.clear()
        self.ax.plot([self.waypoints[waypoint][0] for waypoint in path],
                     [self.waypoints[waypoint][1] for waypoint in path], 'o-')
        self.canvas.draw()

    def start_simulation(self):
        for generation in range(100):
            fitness, self.record_efficiency, self.best_path = calculate_fitness(
                self.waypoints, self.population, self.record_efficiency, self.best_path)
            normalized_fitness = normalize_fitness(fitness)
            self.population = next_generation(self.population, normalized_fitness)

            best_path = self.best_path
            self.root.after(50, self.update_plot, best_path)  # Update plot every 50 milliseconds


if __name__ == "__main__":
    root = tk.Tk()
    root.title("AUV Pathfinding Visualization")
    root.geometry("800x600")

    pathfinding_visualizer = AUVPathfindingVisualizer(root, total_waypoints=10, population_size=50)
    root.mainloop()
