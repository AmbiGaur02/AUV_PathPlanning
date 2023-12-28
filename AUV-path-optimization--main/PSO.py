import tkinter as tk
from tkinter import messagebox
import random
import numpy as np
import matplotlib.pyplot as plt

class AUVParticle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_fitness = float('inf')

def AUV_PSO(PathObjective, Population_size, Dimensions, MaxIterations):
    global_best_position = None
    global_best_fitness = float('inf')
    particles = []
    convergence_history = []

    # Population Initialization
    for _ in range(Population_size):
        position = np.random.uniform(-0.5, 0.5, Dimensions)
        particle = AUVParticle(position)
        particles.append(particle)

        # Fitness Update
        fitness = PathObjective(position)
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = position

        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = position

    # PSO Main Loop
    inertia_weight = 0.8
    cognitive_coefficient = 1.2
    social_coefficient = 1.2

    for itr in range(MaxIterations):
        convergence_history.append(global_best_fitness)

        for particle in particles:
            r1 = random.random()
            r2 = random.random()

            # Velocity Calculation
            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_coefficient * r1 * (particle.best_position - particle.position) +
                                 social_coefficient * r2 * (global_best_position - particle.position))

            # New Position
            particle.position += particle.velocity

            # Evaluate Fitness
            fitness = PathObjective(particle.position)

            # Update Personal Best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            # Update Global Best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position

    # Plot Convergence Graph
    plt.plot(convergence_history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("Convergence Graph")
    plt.show()

    return global_best_position, global_best_fitness

# Define PathObjective functions
def DistanceToTarget(x):
    target_point = np.array([1.0, 1.0])
    return np.sqrt(np.sum((x - target_point)**2))

def MaximumDepth(x):
    return np.max(x)

AUV_Objectives = {'DistanceToTarget': DistanceToTarget, 'MaximumDepth': MaximumDepth}

class AUVPSOGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("AUV PSO GUI")
        self.geometry("400x200")

        self.label = tk.Label(self, text="Select AUV Path Objective:")
        self.label.pack(pady=10)

        self.objective_var = tk.StringVar()
        self.objective_var.set("DistanceToTarget")  # Default objective function

        self.objective_menu = tk.OptionMenu(self, self.objective_var, *AUV_Objectives.keys())
        self.objective_menu.pack(pady=10)

        self.run_button = tk.Button(self, text="Run AUV PSO", command=self.run_auv_pso)
        self.run_button.pack(pady=10)

    def run_auv_pso(self):
        objective_name = self.objective_var.get()
        PathObjective = AUV_Objectives[objective_name]

        Population_size = 50
        MaxIterations = 50
        Dimensions = 2

        best_position, best_fitness = AUV_PSO(PathObjective, Population_size, Dimensions, MaxIterations)

        output_msg = f"Running AUV Path Objective: {objective_name}\n"
        output_msg += f"Best Position: {best_position}\n"
        output_msg += f"Best Fitness: {best_fitness}\n"

        try:
            messagebox.showinfo("AUV PSO RUN", output_msg)
        except tk.TclError:
            # Handle the case where the code is run in a non-GUI environment
            print(output_msg)

if __name__ == "__main__":
    app = AUVPSOGUI()
    app.mainloop()
