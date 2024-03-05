import numpy as np
import networkx as nx
from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt

class MaxKColorGenerator:
    @staticmethod
    def generate(seed, number_of_nodes=20, max_connections_per_node=4, max_colors=None):

        """
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_hiive.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        """
        np.random.seed(seed)
        # all nodes have to be connected, somehow.
        node_connection_counts = np.random.randint(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                      n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]
        problem = MaxKColorOpt(edges=edges, length=number_of_nodes, max_colors=max_colors, source_graph=g)
        return problem


class MaxKColorOpt(DiscreteOpt):
    def __init__(self, edges=None, length=None, fitness_fn=None, maximize=False,
                 max_colors=None, crossover=None, mutator=None, source_graph=None):

        if (fitness_fn is None) and (edges is None):
            raise Exception("fitness_fn or edges must be specified.")

        if length is None:
            if fitness_fn is None:
                length = len(edges)
            else:
                length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = MaxKColor(edges)

        # set up initial state (everything painted one color)
        if source_graph is None:
            g = nx.Graph()
            g.add_edges_from(edges)
            self.source_graph = g
        else:
            self.source_graph = source_graph

        fitness_fn.set_graph(self.source_graph)
        # if none is provided, make a reasonable starting guess.
        # the max val is going to be the one plus the maximum number of neighbors of any one node.
        if max_colors is None:
            total_neighbor_count = [len([*self.source_graph.neighbors(n)]) for n in range(length)]
            max_colors = 1 + max(total_neighbor_count)
        self.max_val = max_colors

        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_colors, crossover, mutator)

        # state = [len([*g.neighbors(n)]) for n in range(length)]
        state = np.random.randint(max_colors, size=self.length)
        np.random.shuffle(state)
        # state = [0] * length
        self.set_state(state)

    def can_stop(self):
        return int(self.get_fitness()) == 0


class MaxKColor:
    """Fitness function for Max-k color optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
    represents the color of node i, as the number of pairs of adjacent nodes
    of the same color.

    Parameters
    ----------
    edges: list of pairs
        List of all pairs of connected nodes. Order does not matter, so (a, b)
        and (b, a) are considered to be the same.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_hiive.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        3

    Note
    ----
    The MaxKColor fitness function is suitable for use in discrete-state
    optimization problems *only*.

    This is a cost minimization problem: lower scores are better than
    higher scores. That is, for a given graph, and a given number of colors,
    the challenge is to assign a color to each node in the graph such that
    the number of pairs of adjacent nodes of the same color is minimized.
    """

    def __init__(self, edges):

        # Remove any duplicates from list
        edges = list({tuple(sorted(edge)) for edge in edges})

        self.graph_edges = None
        self.edges = edges
        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = 0
        for edge in self.edges:
            node1, node2 = edge
            if state[node1] == state[node2]:
                fitness += 1
        """
        if fitness == 0:
            for i in range(len(edges)):
                # Check for adjacent nodes of the same color
                n1, n2 = edges[i]
                print(f'i:{i}: ({n1},{n2})[{state[n1]}] <-> [{state[n2]}]')
        """
        """
        
        if self.graph_edges is not None:
            fitness = sum(int(state[n1] == state[n2]) for (n1, n2) in self.graph_edges)
        else:
            fitness = 0
            for i in range(len(self.edges)):
                # Check for adjacent nodes of the same color
                if state[self.edges[i][0]] == state[self.edges[i][1]]:
                    fitness += 1
        """
        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type

    def set_graph(self, graph):
        self.graph_edges = [e for e in graph.edges()]