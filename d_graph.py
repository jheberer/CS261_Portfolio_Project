# Course: CS261 - Data Structures
# Author: Jonathan Heberer
# Assignment: 6
# Description: Implementation of directed list

import heapq
from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Add vertex to the graph and return the vertex count
        """
        # increase the length of every list within the matrix by one,
        # which will have a weight of 0
        for row in self.adj_matrix:
            row.append(0)

        # increase the vertex count (doing now makes the next step easier)
        self.v_count += 1

        # add an empty list to the adj_matrix, with number of elements equal
        # to the number of vertices, and having weight of 0 to all other vertices
        self.adj_matrix.append(self.v_count*[0])

        return self.v_count


    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Add edge between two vertices. src and dst are indices for vertices,
        and weight is the desired weight between them.
        """
        if src >= self.v_count or src < 0:
            return

        if dst >= len(self.adj_matrix[src]) or dst < 0:
            return

        if src == dst:
            return

        if weight <= 0:
            return

        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Remove directed edge from src to dst.

        Do nothing if edge or vertices do not exist.
        """
        if src >= self.v_count or src < 0:
            return

        if dst >= len(self.adj_matrix[src]) or dst < 0:
            return

        # don't need to check if the edge weight is 0--same effect either way
        self.adj_matrix[src][dst] = 0


    def get_vertices(self) -> []:
        """
        Returns a list containing the vertices in the graph.
        """
        vertex_list = [i for i in range(self.v_count)]

        return vertex_list


    def get_edges(self) -> []:
        """
        Returns a list of the edges in the graph. Each edge is a tuple of two incident vertex
        indices and the weight of the edge.

        First element: source vertex
        Second element: destination vertex
        Third element: Weight of edge.
        """
        # iterate over each vertex (row) in the matrix
        # keep the index of the vertex and each column (destination vertex)
        # if the value of the column is not 0, make a tuple
        # add tuple to the list

        edge_list = []
        for vertex_i in range(self.v_count):
            # v_count works again because the # of stored edges == v_count
            for edge_i in range(self.v_count):
                curr_edge_weight = self.adj_matrix[vertex_i][edge_i]
                if curr_edge_weight != 0:
                    edge_list.append((vertex_i, edge_i, curr_edge_weight))

        return edge_list

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list as an input of vertex indices and determines if they represent
        a valid path.
        """
        # empty path is valid
        if not path:
            return True

        # path requires unique visits to vertices
        if len(path) > self.v_count:
            return False

        # index out of range:
        for i in range(len(path)):
            if i >= self.v_count:
                return False

        # path is one vertex
        if len(path) == 1:
            if path[0] < self.v_count:
                return True
            else:
                return False

        # now analyze path

        i = 0

        while i < len(path) - 1:
            curr_vertex_index = path[i]
            next_vertex_index = path[i+1]

            if self.adj_matrix[curr_vertex_index][next_vertex_index] == 0:
                return False

            i += 1

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Perform a depth first search of the graph. Return a list of vertices visited
        and in the order they are visited.

        v_start is the vertex index to start at and v_end (if it exists)
        will stop the search (inclusive)
        """
        visited_vertices = []

        # start vertex not in graph
        if v_start < 0 or v_start >= self.v_count:
            return visited_vertices

        # end vertex not in graph
        if v_end is not None:
            if v_end < 0 or v_end >= self.v_count:
                v_end = None

        # use a stack
        # when there's a tie for next vertex, use numerical order of indices
        travel_stack = [v_start]

        while travel_stack:
            curr_vertex_index = travel_stack.pop()

            if curr_vertex_index not in visited_vertices:
                visited_vertices.append(curr_vertex_index)

            if curr_vertex_index == v_end:
                break

            # get direct successors
            curr_direct_successors = []
            for edge_index in range(self.v_count):
                curr_edge_weight = self.adj_matrix[curr_vertex_index][edge_index]
                if curr_edge_weight != 0 and edge_index not in visited_vertices:
                    curr_direct_successors.append(edge_index)

            # we are going to 'pop', so reverse sort makes us pop in ascending order
            curr_direct_successors.sort(reverse=True)
            travel_stack.extend(curr_direct_successors)

        return visited_vertices


    def bfs(self, v_start, v_end=None) -> []:
        """
        Perform breadth first search of graph. Return a list of the vertices
        in the order in which they were visited.

        v_start is the vertex index to start at and v_end (if it exists)
        will stop the search (inclusive)
        """
        visited_vertices = []

        # start vertex not in graph
        if v_start < 0 or v_start >= self.v_count:
            return visited_vertices

        # end vertex not in graph
        if v_end is not None:
            if v_end < 0 or v_end >= self.v_count:
                v_end = None

        # use a queue for BFS
        travel_deque = deque()
        travel_deque.append(v_start)

        while travel_deque:
            curr_vertex_index = travel_deque.popleft()

            if curr_vertex_index not in visited_vertices:
                visited_vertices.append(curr_vertex_index)

            # if we reach the end vertex, break early
            if curr_vertex_index == v_end:
                break

            # get direct successors

            curr_direct_successors = []
            for edge_index in range(self.v_count):
                curr_edge_weight = self.adj_matrix[curr_vertex_index][edge_index]
                if curr_edge_weight != 0 and edge_index not in visited_vertices:
                    curr_direct_successors.append(edge_index)

            # we are going to 'pop', so reverse sort makes us pop in ascending order
            curr_direct_successors.sort()
            travel_deque.extend(curr_direct_successors)

        return visited_vertices


    def has_cycle(self):
        """
        Return True if graph contains a cycle, false if it's acyclic
        """
        # Can try something similar to what I did for UD graphs, and use DFS
        # and look for back edges.

        # iterate over all indices as start in case graph is not connected
        for v_start_index in range(self.v_count):

            # track vertices and parents
            vertex_parent_dict = {}
            for vertex_index in range(self.v_count):
                vertex_parent_dict[vertex_index] = None

            visited_vertices = []
            dead_end_vertices = []

            # do some DFS magic here
            travel_stack = [v_start_index]

            while travel_stack:
                curr_vertex_index = travel_stack.pop()

                if curr_vertex_index not in visited_vertices:
                    visited_vertices.append(curr_vertex_index)

                # get direct successors
                curr_direct_successors = []
                i = 0
                for edge_index in range(self.v_count):
                    curr_edge_weight = self.adj_matrix[curr_vertex_index][edge_index]
                    if curr_edge_weight != 0:
                        if edge_index not in visited_vertices:
                            curr_direct_successors.append(edge_index)
                            vertex_parent_dict[edge_index] = curr_vertex_index
                        elif edge_index != vertex_parent_dict[curr_vertex_index]:
                            # if the next vertex ALSO points back to an already visit vertex
                            for i in range(self.v_count):
                                weight = self.adj_matrix[edge_index][i]
                                if weight != 0:
                                    if i in visited_vertices and i not in dead_end_vertices:
                                        return True
                    else:
                        i += 1
                    if i == self.v_count:
                        dead_end_vertices.append(curr_vertex_index)

                # we are going to 'pop', so reverse sort makes us pop in ascending order
                curr_direct_successors.sort(reverse=True)
                travel_stack.extend(curr_direct_successors)

        return False


    def dijkstra(self, src: int) -> []:
        """
        Find the length of the shortest path from a given vertex to all
        other vertices. The output is a list where each index represents
        the respective vertex, and the value at index represents the distance
        from the src to that vertex. If vertex is not reachable then the value is
        float('inf'). Assume src is valid input.
        """
        # use hash table for visited vertices per the exploration
        visited_vertices = dict()
        for i in range(self.v_count):
            visited_vertices[i] = float('inf')

        # empty priority queue for travel
        travel_heapq = []
        heapq.heappush(travel_heapq, (0, src))

        while travel_heapq:
            v = heapq.heappop(travel_heapq)

            # first index element of v represents the vertex
            v_vertex_index = v[1]
            v_distance = v[0]

            if visited_vertices[v_vertex_index] == float('inf'):
                visited_vertices[v_vertex_index] = v_distance

            # get direct successors

            curr_direct_successors = []
            for edge_index in range(self.v_count):
                curr_edge_weight = self.adj_matrix[v_vertex_index][edge_index]
                if curr_edge_weight != 0 and visited_vertices[edge_index] == float('inf'):
                    curr_direct_successors.append((curr_edge_weight + v_distance, edge_index))

            # we are going to 'pop', so reverse sort makes us pop in ascending order
            curr_direct_successors.sort()
            # for v in curr_direct_successors:
            #     heapq.heappush(travel_heapq, curr_direct_successors)
            travel_heapq.extend(curr_direct_successors)
            heapq.heapify(travel_heapq)

        visited_vertices_list = [None] * self.v_count
        for key, item in visited_vertices.items():
            visited_vertices_list[key] = item

        return visited_vertices_list


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')

    print("\nPDF - method has_cycle() custom 2")
    print("----------------------------------")
    edges = [(0, 5, 11), (0, 6, 18), (1, 10, 14), (2, 9, 10), (3, 1, 8), (4, 7, 3), (5, 8, 8), (6, 5, 17), (6, 11, 12), (11, 1, 20), (11, 3, 16), (12, 6, 1)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    # print("\nPDF - method has_cycle() custom 1")
    # print("----------------------------------")
    # edges = [(0, 1, 10), (1, 3, 1), (1, 4, 15), (2, 1, 23), (2, 3, 1), (4, 3, 1)]
    # g = DirectedGraph(edges)
    # print(g.get_edges(), g.has_cycle(), sep='\n')
    # print('\n', g)
    #
    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0, 99)]
    for src, dst, *weight in edges_to_add:
        g.add_edge(src, dst, *weight)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    # print("\nPDF - dijkstra() example 1")
    # print("--------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # for i in range(5):
    #     print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    # g.remove_edge(4, 3)
    # print('\n', g)
    # for i in range(5):
    #     print(f'DIJKSTRA {i} {g.dijkstra(i)}')
