# -*- coding: utf-8 -*-

'''
Pacman Project
    
Autors:
- Bruna Luisa Poffo Nobre,
- Fernando Rodrigues Santos,
- Iuri Sônego Cardoso,
- Thais Juliane Dall’Agnol

Copy from Jomi Hübner(https://jomifred.github.io/ia/) AI code(https://colab.research.google.com/drive/1tylaeC-A29rvoCU1O7S5E5EAwCT-Zt9N?usp=sharing) modified from [AIMA code](https://github.com/aimacode/aima-python/blob/master/search4e.ipynb)

We added

- Pacman problem

and modified some of the original search algorithms:

- best_first_search;
- best_first_tree_search;
- astar_search;
- greedy_bfs;
- breadth_first_bfs;
- depth_first_bfs;
- breadth_first_search.
    
'''

# ------------------------------------------------------------------------------
# Problems and Nodes
# ------------------------------------------------------------------------------

# matplotlib inline
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
import time
from collections import defaultdict, deque, Counter
from itertools import combinations


class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal` states
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1
    def h(self, node):               return 0

    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)


class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.


def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


# ------------------------------------------------------------------------------
# Queues
# ------------------------------------------------------------------------------

FIFOQueue = deque

LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)


# ------------------------------------------------------------------------------
# Search Algorithms: Best-First
# ------------------------------------------------------------------------------

def best_first_search(problem, f, processingCostResult=None):
    "Busca nós com o valor mínimo de f(nó) primeiro"

    def setProcessingCostResult(processingCostResult, startTime, visitedNodes):
        if processingCostResult != None:
            processingCostResult["time"] = time.perf_counter() - startTime
            processingCostResult["visited nodes"] = visitedNodes
        return

    startTime = time.perf_counter()
    visitedNodes = 1
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}

    while frontier:
        node = frontier.pop()
        visitedNodes += 1
        if problem.is_goal(node.state):
            setProcessingCostResult(processingCostResult, startTime, visitedNodes)
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    setProcessingCostResult(processingCostResult, startTime, visitedNodes)
    return failure


def best_first_tree_search(problem, f):
    "Uma versão de 'best_first_search' sem a tabela 'alcançada'"
    frontier = PriorityQueue([Node(problem.initial)], key=f)
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            if not is_cycle(child):
                frontier.add(child)
    return failure


def g(n):
    return n.path_cost


def astar_search(problem, h=None, processingCostResult=None):
    """Busca nós com mínimo f(n) = g(n) + h(n)"""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n), processingCostResult=processingCostResult)


def astar_tree_search(problem, h=None):
    """Busca nós com mínimo f(n) = g(n) + h(n), sem tabela 'alcançada'"""
    h = h or problem.h
    return best_first_tree_search(problem, f=lambda n: g(n) + h(n))


def weighted_astar_search(problem, h=None, weight=1.4):
    """Busca nós com mínimo f(n) = g(n) + peso * h(n)"""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + weight * h(n))


def greedy_bfs(problem, h=None, processingCostResult=None):
    """Busca nós com h(n) mínimo"""
    h = h or problem.h
    return best_first_search(problem, f=h, processingCostResult=processingCostResult)


def uniform_cost_search(problem):
    "Busca primeiro os nós com menor custo de caminho"
    return best_first_search(problem, f=g)


def breadth_first_bfs(problem, processingCostResult=None):
    "Busca primeiro os nós mais superficiais na árvore de pesquisa; usando o melhor primeiro"
    return best_first_search(problem, f=len, processingCostResult=processingCostResult)


def depth_first_bfs(problem, processingCostResult=None):
    "Busca primeiro os nós mais profundos na árvore de pesquisa; usando o melhor primeiro"
    return best_first_search(problem, f=lambda n: -len(n), processingCostResult=processingCostResult)


def is_cycle(node, k=30):
    "Este nó forma um ciclo de comprimento k ou menor?"
    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))

# ------------------------------------------------------------------------------
# Other Search Algorithms
# ------------------------------------------------------------------------------

def breadth_first_search(problem, processingCostResult=None):
    "Busca primeiro os nós mais superficiais na árvore de busca"    

    def setProcessingCostResult(processingCostResult, startTime, visitedNodes):
        if processingCostResult != None:
            processingCostResult["time"] = time.perf_counter() - startTime
            processingCostResult["visited nodes"] = visitedNodes
        return

    startTime = time.perf_counter()
    visitedNodes = 1
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        setProcessingCostResult(processingCostResult, startTime, visitedNodes)
        return node
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        visitedNodes += 1
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                setProcessingCostResult(processingCostResult, startTime, visitedNodes)
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    
    setProcessingCostResult(processingCostResult, startTime, visitedNodes)
    return failure


def iterative_deepening_search(problem):
    "Faz uma busca com profundidade limitada com limites de profundidade crescentes"
    for limit in range(1, sys.maxsize):
        result = depth_limited_search(problem, limit)
        if result != cutoff:
            return result


def depth_limited_search(problem, limit=10):
    "Busca primeiro os nós mais profundos na árvore de busca"
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        elif len(node) >= limit:
            result = cutoff
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
    return result


def depth_first_recursive_search(problem, node=None):
    if node is None:
        node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node
    elif is_cycle(node):
        return failure
    else:
        for child in expand(problem, node):
            result = depth_first_recursive_search(problem, child)
            if result:
                return result
        return failure


# ------------------------------------------------------------------------------
# Pacman problem
# ------------------------------------------------------------------------------


class PacmanProblem(Problem):
    """
    state is the map of positions with pacman, food, obstacle and empty:
      Example map (rows and columns):
        map = (
          "---",
          "-P-",
          "---"
          )
    """

    PACMAN = "P"
    FOOD = "-"
    PACMAN_AND_FOOD = "#"
    OBSTACLE = "X"
    EMPTY = " "
    GHOST = "F"

    """ Trasitions: move the pacman up, down, left or right """
    ACTION_UP = (0, -1)
    ACTION_LEFT = (-1, 0)
    ACTION_RIGHT = (+1, 0)
    ACTION_DOWN = (0, +1)

    def __init__(self, initial, goal=None, **kwds):
        super().__init__(initial)

    def actions(self, state):
        actions = set()
        px, py = self.getPacmanPosition(state)
        if py > 0 and state[py - 1][px] != self.OBSTACLE:
            actions.add(self.ACTION_UP)

        if py < len(state) - 1 and state[py + 1][px] != self.OBSTACLE:
            actions.add(self.ACTION_DOWN)

        if px > 0 and state[py][px - 1] != self.OBSTACLE:
            actions.add(self.ACTION_LEFT)

        if px < len(state[py]) - 1 and state[py][px + 1] != self.OBSTACLE:
            actions.add(self.ACTION_RIGHT)

        return actions

    def result(self, state, action):
        """'Executa a ação e retorna o novo estado"""
        px, py = self.getPacmanPosition(state)

        newpx = px + action[0]
        newpy = py + action[1]

        assert (
            newpx == min(max(newpx, 0), len(state[py]) - 1)
            or newpy == min(max(newpy, 0), len(state) - 1)
            or state[newpy][newpx] == self.OBSTACLE
        ), f"Invalid action {action} for state {state}"

        newState = []
        for line in state:
            newState.append(line)

        pacman = (
            self.PACMAN_AND_FOOD if state[newpy][newpx] == self.FOOD else self.PACMAN
        )

        newState[newpy] = (
            newState[newpy][:newpx] + pacman + newState[newpy][newpx + 1 :]
        )
        newState[py] = newState[py][:px] + self.EMPTY + newState[py][px + 1 :]

        return tuple(newState)

    def is_goal(self, state):
        """Testa se ainda tem comida no map, retorna 'False' se ainda tiver e retorna 'True' quando o mapa está limpo"""
        for line in state:
            if line.find(self.FOOD) >= 0:
                return False
        return True

    def getPacmanPosition(self, state):
        """Retorna a posição do PacMan"""
        qlines = len(state)
        qcols = len(state[0])
        for y in range(qlines):
            x = state[y].find(self.PACMAN)
            if x != -1:
                return (x, y)
            x = state[y].find(self.PACMAN_AND_FOOD)
            if x != -1:
                return (x, y)
        raise ValueError(f"Invalid State: cannot find PACMAN in {state}")

    def getNearestFoodDistance(self, state):
        """Retorna a distância de Manhattan da(s) comida(s) mais próxima(s) do do PACMAN"""

        def isValidPosition(x, y, w, l):
            """Sub-função para verificar se uma coordenada é valida."""
            return x >= 0 and x < w and y >= 0 and y < l

        qlines = len(state)
        qcols = len(state[0])
        px, py = self.getPacmanPosition(state)

        # se tem comida na posição do PACMAN, já encontrou.
        if state[py][px] == self.PACMAN_AND_FOOD:
            return 0

        # Faz a "Procura Quadrada" por comida em volta do PACMAN (considerando distância de Manhattan).
        for dist in range(1, qlines + qcols):
            for i in range(0, dist):
                # busca da esquerda para cima
                if (
                    isValidPosition(px - dist + i, py + i, qcols, qlines)
                    and state[py + i][px - dist + i] == self.FOOD
                ):
                    return dist

                # busca de cima para a direita
                if (
                    isValidPosition(px + i, py + dist - i, qcols, qlines)
                    and state[py + dist - i][px + i] == self.FOOD
                ):
                    return dist

                # busca da direita para baixo
                if (
                    isValidPosition(px + dist - i, py - i, qcols, qlines)
                    and state[py - i][px + dist - i] == self.FOOD
                ):
                    return dist

                # busca de baixo para a esquerda
                if (
                    isValidPosition(px - i, py - dist + i, qcols, qlines)
                    and state[py - dist + i][px - i] == self.FOOD
                ):
                    return dist

        # Não há comida no mapa.
        raise ValueError(
            "ERROR: There are no food on the map. Was the objective achieved?"
        )

    def getFurthestFoodPositionInNearestSector(self, state):

        def isValidPosition(x, y, w, l):
            """Sub-função para verificar se uma coordenada é valida."""
            return x >= 0 and x < w and y >= 0 and y < l

        qlines = len(state)
        qcols = len(state[0])
        px, py = self.getPacmanPosition(state)

        # calcula os valores de referência para a diagonais principal e secundária.
        diag1 = px - py  # principal
        diag2 = px + py  # secundária

        # conjuntos das comidas encontradas em cada setor.
        up = -1
        right = -1
        down = -1
        left = -1

        cornersDistance = [
            manhattanDistance((px,py), (0,0)),
            manhattanDistance((px,py),(0, qlines-1)),
            manhattanDistance((px,py),(qcols-1, 0)),
            manhattanDistance((px,py),(qcols-1, qlines-1)),
        ]

        # Faz a "Procura Quadrada" por comida em volta do PACMAN (considerando distância de Manhattan).
        for dist in range(max(cornersDistance),0,-1):
            for i in range(0, dist+1):
                # busca da esquerda para baixo
                x = px - dist + i
                y = py + i
                if (
                    isValidPosition(x, y, qcols, qlines)
                    and state[y][x] == self.FOOD
                ):
                    # Faz os cálculos de referência da posição da comida para comparação com as diagonais
                    calc2 = x + y

                    if calc2 >= diag2 and down == -1:
                        # adiciona a distância no setor inferior
                        down = dist                    
                    if calc2 <= diag2 and left == -1:
                        # adiciona a distância no setor esquerdo
                        left = dist

                # busca de baixo para a direita
                x = px + i
                y = py + dist - i
                if (
                    isValidPosition(x, y, qcols, qlines)
                    and state[y][x] == self.FOOD
                ):
                    # Faz os cálculos de referência da posição da comida para comparação com as diagonais
                    calc1 = x - y

                    if calc1 <= diag1 and down == -1:
                        # adiciona a distância no setor inferior
                        down = dist                        
                    if calc1 >= diag1 and right == -1:
                        # adiciona a distância no setor direito
                        right = dist

                # busca da direita para cima
                x = px + dist - i
                y = py - i
                if (
                    isValidPosition(x, y, qcols, qlines)
                    and state[y][x] == self.FOOD
                ):
                    # Faz os cálculos de referência da posição da comida para comparação com as diagonais
                    calc2 = x + y

                    if calc2 >= diag2 and right == -1:
                        # adiciona a distância no setor direito
                        right = dist                        
                    if calc2 <= diag2 and up == -1:
                        # adiciona a distância no setor superior
                        up = dist

                # busca de cima para a esquerda
                x = px - i
                y = py - dist + i
                if (
                    isValidPosition(x, y, qcols, qlines)
                    and state[y][x] == self.FOOD
                ):
                     # Faz os cálculos de referência da posição da comida para comparação com as diagonais
                    calc1 = x - y

                    if calc1 >= diag1 and up == -1:
                        # adiciona a distância no setor superior
                        up = dist
                    if calc1 <= diag1 and left == -1:
                        # adiciona a distância no setor esquerdo
                        left = dist    

                # Se já encontrou os últimos de todos os setores, finaliza este quadrado
                if up != -1 and right != -1 and down != -1 and left != -1:
                    break
            # Se a busca no quadrado foi até o fim e ainda precisa continuar a busca, segue para a próxima distância. 
            else:
                continue

            # Se chegou aqui, o FOR anterior foi interrompido, então já encontrou todas as distâncias. Finaliza!
            break

        # lista dos setores com comida encontrada.
        sectorsWithFood = []

        for s in [up, right, left, down]:
            if s != -1:
                sectorsWithFood.append(s)

        # Se não encontrou comida ...
        if len(sectorsWithFood) == 0:

            # se o PACMAN está sobre a última comida, retorna zero.
            if state[py][px] == self.PACMAN_AND_FOOD:
                return 0
            
            # senão, erro!
            raise ValueError(
                "ERROR: There are no food on the map. Was the objective achieved?"
            )

        # Retorna a posição da comida mais longe do setor mais próximo.
        return min(sectorsWithFood)

    def action_cost(self, stateCurrent, action, stateTarget):
        return 1

    def h(self, node):
        """Heurística baseada na Distância de Manhattan"""
        return self.getNearestFoodDistance(node.state)

    def h2(self, node):
        """Heurística baseada na Distância de Manhattan do setor de comidas mais próximo"""
        return self.getFurthestFoodPositionInNearestSector(node.state)


def manhattanDistance(A, B):
    """Cálculo da Distância de Manhattan entre dois pontos A e B"""
    return abs(A[0] - B[0]) + abs(A[1] - B[1])


# ------------------------------------------------------------------------------
# Solution
# ------------------------------------------------------------------------------


def h1(problem):
    return problem.h


def h2(problem):
    return problem.h2


# Map Class
class Map:
    def __init__(self, map, desc=None):
        assert isinstance(map, tuple), "ERROR: map should be a tuple."
        assert len(map) > 0, "ERROR: map cannot be empty."
        width = len(map[0])
        length = 0
        for line in map:
            assert (
                len(line) == width
            ), f"ERROR: map should be regular (square or retangle). Line {length} has different width."
            length += 1

        self.__map = map
        self.__width = width
        self.__length = length

        if desc == None:
            desc = f"{width}x{length}"
        self.__desc = desc

    def getDimensions(self):
        return [self.__width, self.__length]

    def getMap(self):
        return self.__map

    def __str__(self):
        string = self.__desc + "\n"
        for line in self.__map:
            string += line + "\n"
        return string
    
    def getDescription(self):
        return self.__desc


# Create map list
maps = []

map5x5 = ("-    ", " -X  ", "XXP  ", " -X- ", "    -")

map7x7 = ("-      ", " XX-XX-", " X   X ", "- -P   ", " X  -X ", " XX XX ", "  -   -")

map10x10 = (
    "    -     ",
    " XXX  XXX ",
    "       -  ",
    "  - XX    ",
    "    XX    ",
    "  X P     ",
    "  X  X - X",
    "  X  X   X",
    "  X- X    ",
    "     X  X-",
)

map15x15 = (
    "-X     X      -",
    " X     X       ",
    " X X X X X X X ",
    "               ",
    "XXXXXXXX   XXXX",
    "    -   X      ",
    "  XXXXX XXXXX  ",
    "    X  P       ",
    "     X         ",
    "      X        ",
    "       X       ",
    "        X      ",
    "         X     ",
    "          X    ",
    "-             -",
)

map20x20 = (
    "-                  -",
    " X X X X X X X X X X",
    "  X              X  ",
    "   X            X   ",
    "    X    -     X    ",
    "    -X        X     ",
    "      X      X      ",
    "       X    X       ",
    "        X  X        ",
    "         P XX  X  X-",
    "        X X         ",
    "       X   X        ",
    "      X     X       ",
    "     X       X      ",
    "    X-        X     ",
    "   X          -X    ",
    " -X             X   ",
    " X               X  ",
    "X                 X ",
    "-                  -",
)


# ----- Mapas para verificação de possíveis vantagens na heurística H2 -----

map3X20_tH2 = (
    "                    ",
    "- - - - -P     -    ",
    "                    "
)

map5x20_tH2 = (
    "              -     ",
    "              -     ",
    "---------P    -     ",
    "              -     ",
    "          -   -     "
)

map17x20_tH2 = (
    " --      ----   ----",
    "----     ----    ---",
    "-----    ---     ---",
    " --      ---      --",
    "                    ",
    "               -----",
    "         P     -----",
    "               -----",
    "               -----",
    "                    ",
    "                    ",
    "                    ",
    "                    ",
    "                    ",
    "-                   ",
    "--         -----    ",
    "---        ------   "
)

map20x20_tH2 = (
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "--------------------",
    "------------------- ",
    "------------------  ",
    "-----------------   ",
    "----------------  --",
    "---------------   --",
    "--------------P   --",
    "-------------     --",
    "------------ ---  --",
    "-----------  ---    "
)

# --------------------------------------------------------------------------

def statelist_to_string(states, cols=80):
    n_states = len(states)
    n_lines = len(states[0])
    n_cols = len(states[0][0]) + 1
    statesInLine = cols // n_cols
    
    output = "_" * n_cols * statesInLine + "\n"
    for iS in range(0, n_states, statesInLine):
        for iI in range(n_lines):
            for iS2 in range(iS, min(iS+statesInLine,n_states)):
                output += states[iS2][iI] + "|"
            output += "\n"
        output += "_" * n_cols * statesInLine + "\n"
    return output


""" Testes """
""" algoritmo h1 não funcionou conforme esperado. Durante a priorização dos nós, buscando sempre
    a comida mais próxima, ao limpar o primeiro setor (esquerda), os próximos nós ficam com custo estimado
    (h) grandes, priorizando outros nós que antes haviam sido desprezados (prioridade baixa). Ao final, o
    algoritmo h1 acaba priorizando o setor com lado oposto mais próximo, selecionado o mesmo caminho de h2.
"""
""" pac = PacmanProblem(map5x20_tH2)
path = greedy_bfs(pac, pac.h)
actions = path_actions(path)
states = path_states(path)
string = statelist_to_string(states)
print("Algorithm: greedy_bfs, Heuristic: H1")
print("Actions: ",actions)
print("States: ", string)


path = greedy_bfs(pac, pac.h2)
actions = path_actions(path)
states = path_states(path)
string = statelist_to_string(states)
print("\nAlgorithm: greedy_bfs, Heuristic: H2")
print("Actions: ",actions)
print("States: ", string)

print("=" * 80) """


#maps.append(Map(map5x5))
#maps.append(Map(map7x7))
#maps.append(Map(map10x10))
#maps.append(Map(map15x15))
#maps.append(Map(map20x20))

maps.append(Map(map3X20_tH2, "20x3 assimétrico"))
maps.append(Map(map5x20_tH2, "20x5 assimétrico"))
#maps.append(Map(map17x20_tH2, "20x17 assimétrico"))
#maps.append(Map(map20x20_tH2, "20x20 assimétrico"))


''' Algoritmos e heurísticas '''
algorithms = [
    #depth_first_bfs,
    #breadth_first_bfs,
    #breadth_first_search,
    (greedy_bfs, h1),
    (greedy_bfs, h2),
    (astar_search, h1),
    (astar_search, h2)
]

data = []

dicionario = {
    h1 : "H1 - Nearest Food",
    h2 : "H2 - Nearest Opposite Border",
    depth_first_bfs : "depth_first_bfs",
    breadth_first_bfs : "breadth_first_bfs",
    breadth_first_search : "breadth_first_search",
    greedy_bfs : "greedy_bfs",
    astar_search : "astar_search"
}

import pandas as pd

with open("output.txt", "w") as arquivo:
    for m in maps:
        for a in algorithms:
            pac = PacmanProblem(m.getMap())
            item = []
            print(f"Map: {m.getDescription()}", end=", ")
            if isinstance(a, tuple):
                print(f"algoritm: {dicionario[a[0]]}, heuristic: {dicionario[a[1]]}")
                processingCostResult = {}
                path = a[0](pac, a[1](pac), processingCostResult)
                actions = path_actions(path)

                item = [
                    f"{m.getDescription()}",
                    dicionario[a[0]],
                    dicionario[a[1]],
                    f"{len(actions)}",
                    processingCostResult["time"].__round__(8),
                    processingCostResult["visited nodes"]
                ]
            else:
                processingCostResult = {}
                path = a(pac, processingCostResult)
                actions = path_actions(path)

                item = [
                    f"{m.getDescription()}",
                    dicionario[a],
                    "None",
                    f"{len(actions)}",
                    processingCostResult["time"].__round__(8),
                    processingCostResult["visited nodes"]
                ]
            string = statelist_to_string(path_states(path))
            data.append(item)
            print(item)
            print(string)
            arquivo.write(f"{item}:\n{string}")

df = pd.DataFrame(data, columns=["Map", "Algorithm", "Heuristic function", "Steps", "Time", "Visited Nodes"])
print("=" * 80)
print("\n", df)
df.to_csv("output.csv")
print("Finished")
