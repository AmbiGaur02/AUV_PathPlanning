import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

class ACO:
    def __init__(self, num_ants, evaporation_rate, total_pheromone, graph):
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.total_pheromone = total_pheromone
        self.graph = graph
        self.pheromone_levels = self.initialize_pheromones()

    def initialize_pheromones(self):
        pheromones = {}
        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                pheromones[(i, j)] = 1.0
        return pheromones

    def desirability(self, i, j):
        energy_consumption = self.calculate_energy_consumption(i, j)
        return 1.0 / (energy_consumption + 1e-10)

    def calculate_energy_consumption(self, i, j):
        distance = self.calculate_distance(i, j)
        energy_consumption = distance * 0.1
        return energy_consumption

    def calculate_distance(self, i, j):
        return abs(i - j)

    def heuristic(self, i, j):
        goal_node = self.get_goal_node()
        distance_to_goal_i = self.calculate_distance(i, goal_node)
        distance_to_goal_j = self.calculate_distance(j, goal_node)
        heuristic_value = 1.0 / (distance_to_goal_i + distance_to_goal_j + 1e-10)
        return heuristic_value

    def get_goal_node(self):
        return len(self.graph) - 1

    def update_pheromones(self, ant_paths):
        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                delta_tau = 0.0
                for path in ant_paths:
                    if (i, j) in zip(path, path[1:]):
                        desirability_factor = self.desirability(i, j)
                        heuristic_factor = self.heuristic(i, j)
                        path_length = self.path_length_function(path)
                        delta_tau += (desirability_factor * heuristic_factor) / path_length
                self.pheromone_levels[(i, j)] = (
                    (1 - self.evaporation_rate) * self.pheromone_levels[(i, j)] + delta_tau
                )

    def ant_movement(self, start_node):
        current_node = start_node
        visited_nodes = [current_node]

        while len(visited_nodes) < len(self.graph):
            possible_moves = [node for node in range(len(self.graph)) if node not in visited_nodes]
            probabilities = self.calculate_move_probabilities(current_node, possible_moves)
            next_node = self.np_choice(possible_moves, 1, p=probabilities)[0]

            visited_nodes.append(next_node)
            current_node = next_node

        return visited_nodes

    def calculate_move_probabilities(self, current_node, possible_moves):
        pheromone_values = [self.pheromone_levels[(current_node, node)] for node in possible_moves]
        total_pheromone = sum(pheromone_values)
        probabilities = [value / (total_pheromone + 1e-10) for value in pheromone_values]
        return probabilities

    @staticmethod
    def path_length_function(path):
        total_length = 0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            total_length += abs(node1 - node2)
        return total_length

    @staticmethod
    def np_choice(a, size, replace=True, p=None):
        idx = np.array(random.sample(a, size))
        return idx

    def run_aco_with_convergence_and_visualization(self, num_iterations):
        convergence_data = []

        pygame.init()

        # Set up the Pygame window
        window_size = 600
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption('ACO Pheromone Visualization')

        clock = pygame.time.Clock()

        font = pygame.font.Font(None, 36)

        for iteration in range(num_iterations):
            ant_paths = []
            for ant in range(self.num_ants):
                start_node = np.random.randint(0, len(self.graph))
                ant_path = self.ant_movement(start_node)
                ant_paths.append(ant_path)

            self.update_pheromones(ant_paths)

            best_path_length = self.calculate_best_path_length(ant_paths)
            convergence_data.append(best_path_length)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.draw_pheromones(screen, window_size)
            self.display_best_path_length(screen, font, best_path_length)
            pygame.display.flip()
            clock.tick(60)

        plt.plot(range(1, num_iterations + 1), convergence_data, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Best Path Length')
        plt.title('ACO Convergence')
        plt.show()

    def calculate_best_path_length(self, ant_paths):
        best_path = min(ant_paths, key=self.path_length_function)
        best_path_length = self.path_length_function(best_path)
        return best_path_length

    def draw_pheromones(self, screen, window_size):
        screen.fill((255, 255, 255))  # White background

        max_pheromone = max(self.pheromone_levels.values())
        normalized_pheromones = {k: v / max_pheromone for k, v in self.pheromone_levels.items()}

        cell_size = window_size / len(self.graph)
        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                pheromone_value = normalized_pheromones[(i, j)]
                color = self.get_color_for_pheromone(pheromone_value)
                pygame.draw.rect(screen, color, (i * cell_size, j * cell_size, cell_size, cell_size))

    def display_best_path_length(self, screen, font, best_path_length):
        text = font.render(f'Best Path Length: {best_path_length:.2f}', True, (0, 0, 0))
        screen.blit(text, (10, 10))

    def get_color_for_pheromone(self, pheromone_value):
        color_ranges = [
            (0, (255, 255, 255)),  # White
            (0.2, (204, 255, 204)),  # Light Green
            (0.4, (102, 255, 102)),  # Green
            (0.6, (0, 128, 0)),  # Dark Green
            (0.8, (0, 102, 0)),  # Forest Green
            (1.0, (0, 51, 0))  # Dark Forest Green
        ]

        for range_max, color in color_ranges:
            if pheromone_value <= range_max:
                return color

        return (0, 0, 0)  # Black

# Example usage:
# Define a graph with energy consumption values between nodes
# graph[i][j] represents the energy consumption from node i to node j
graph = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]])

# Create an instance of ACO with parameters
aco_instance = ACO(num_ants=5, evaporation_rate=0.1, total_pheromone=10, graph=graph)

# Run ACO algorithm for a certain number of iterations
aco_instance.run_aco_with_convergence_and_visualization(num_iterations=50)
