import pygame
from queue import PriorityQueue
import matplotlib.pyplot as plt

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Planning and Optimization")

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
MAROON = (128, 0, 0)
YELLOW = (255, 255, 0)
WHEAT = (255, 231, 186)

class Node:
    def __init__(self, rows, columns, width, TotalRows):
        self.rows = rows
        self.columns = columns
        self.x = rows * width
        self.y = columns * width
        self.color = BLACK
        self.neighbours = []
        self.TotalRows = TotalRows
        self.width = width

    def getPos(self):
        return self.rows, self.columns

    def isVisited(self):
        return self.color == WHITE

    def isOpen(self):
        return self.color == BLUE

    def isBarrier(self):
        return self.color == RED

    def isStart(self):
        return self.color == GREEN

    def isEnd(self):
        return self.color == MAROON

    def Reset(self):
        self.color = BLACK

    def makeVisited(self):
        self.color = WHITE

    def makeOpen(self):
        self.color = BLUE

    def makeBarrier(self):
        self.color = RED

    def makeStart(self):
        self.color = GREEN

    def makeEnd(self):
        self.color = WHEAT

    def makePath(self):
        self.color = YELLOW

    def updateNeighbours(self, grid):
        if self.rows < self.TotalRows - 1 and not grid[self.rows + 1][self.columns].isBarrier():
            self.neighbours.append(grid[self.rows + 1][self.columns])

        if self.rows > 0 and not grid[self.rows - 1][self.columns].isBarrier():
            self.neighbours.append(grid[self.rows - 1][self.columns])

        if self.columns < self.TotalRows - 1 and not grid[self.rows][self.columns + 1].isBarrier():
            self.neighbours.append(grid[self.rows][self.columns + 1])

        if self.columns > 0 and not grid[self.rows][self.columns - 1].isBarrier():
            self.neighbours.append(grid[self.rows][self.columns - 1])

    def __lt__(self, other):
        return False

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

def h(z1, z2):
    x1, y1 = z1
    x2, y2 = z2
    return abs(x2 - x1) + abs(y2 - y1)

def reconstructPath(cameFrom, current, draw):
    while current in cameFrom:
        current.makePath()
        current = cameFrom[current]
        draw()

def AStarAlgorithm(draw, grid, start, end):
    count = 0
    openSET = PriorityQueue()
    openSET.put((0, count, start))
    cameFrom = {}
    gScore = {node: float("inf") for row in grid for node in row}
    gScore[start] = 0
    fScore = {node: float("inf") for row in grid for node in row}
    fScore[start] = h(start.getPos(), end.getPos())
    openSetNode = {start}
    explored_nodes = 0

    while not openSET.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = openSET.get()[2]
        openSetNode.remove(current)
        explored_nodes += 1

        if current == end:
            reconstructPath(cameFrom, end, draw)
            end.makeEnd()
            return True, explored_nodes

        for neighbour in current.neighbours:
            tentativegScore = gScore[current] + 1

            if tentativegScore < gScore[neighbour]:
                cameFrom[neighbour] = current
                gScore[neighbour] = tentativegScore
                fScore[neighbour] = tentativegScore + h(neighbour.getPos(), end.getPos())
                if neighbour not in openSetNode:
                    count += 1
                    openSET.put((fScore[neighbour], count, neighbour))
                    openSetNode.add(neighbour)
                    neighbour.makeOpen()

        draw()
        if current != start:
            current.makeVisited()

    return False, explored_nodes

def makeGrid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

def drawGrid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, MAROON, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, MAROON, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(BLACK)
    for row in grid:
        for node in row:
            node.draw(win)

    drawGrid(win, rows, width)
    pygame.display.update()

def getClickedPOS(pos, width, rows):
    gap = width // rows
    y, x = pos
    row = y // gap
    cols = x // gap
    return row, cols

def display_convergence_graph(iterations, nodes_explored):
    plt.plot(range(1, iterations + 1), nodes_explored)
    plt.xlabel("Iteration")
    plt.ylabel("Nodes Explored")
    plt.title("Convergence Graph")
    plt.show()

def main(win, width):
    ROWS = 50
    grid = makeGrid(ROWS, width)
    run = True
    started = False
    start = None
    end = None
    iteration = 0
    nodes_explored = []

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if started:
                continue
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                rows, cols = getClickedPOS(pos, width, ROWS)
                node = grid[rows][cols]
                if not start and node != end:
                    start = node
                    start.makeStart()
                elif not end and node != start:
                    end = node
                    end.makeEnd()
                elif node != end and node != start:
                    node.makeBarrier()
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                rows, cols = getClickedPOS(pos, width, ROWS)
                node = grid[rows][cols]
                node.Reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.updateNeighbours(grid)

                    iteration += 1
                    success, explored_nodes = AStarAlgorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    nodes_explored.append(explored_nodes)

                    if success:
                        print(f"Iteration {iteration}: Path found with {explored_nodes} nodes explored")
                    else:
                        print(f"Iteration {iteration}: Path not found")

                    if iteration >= 2:  # Display convergence graph after at least 2 iterations
                        display_convergence_graph(iteration, nodes_explored)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = makeGrid(ROWS, width)
                    iteration = 0
                    nodes_explored = []

        draw(win, grid, ROWS, width)
        pygame.display.update()

    pygame.quit()

# Call the main function
main(WIN, WIDTH)