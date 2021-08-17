# Course: CS 261 Data Structures
# Author: Jonathan Heberer
# Assignment: 6
# Description: Implementation of undirected graph using adjacency list

from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph. If key vertex already exists, do nothing.

        :param v: the name (and key) of the vertex
        """
        if v in self.adj_list:
            return None
        self.adj_list[v] = []
        
    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph. The edge is defined only by the two vertices
        it connects, u and v.

        :param u: a vertex connected by the newly added edge
        :param v: a vertex connected by the newly added edge
        """
        # if u and v are the same vertex, do nothing
        if u == v:
            return

        # if u does not exist yet
        if u not in self.adj_list:
            self.add_vertex(u)

        # if v does not exist yet
        if v not in self.adj_list:
            self.add_vertex(v)

        # if there's already an edge between them, do nothing
        if v in self.adj_list[u]:
            return

        # create edge
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
        

    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph

        :param v: one vertex connected by the edge to be removed
        :param u: one vertex connected by the edge to be removed
        """

        # if either vertex does not exist
        # do nothing
        if v not in self.adj_list or u not in self.adj_list:
            return

        # if the edge does not exist
        # do nothing
        if v not in self.adj_list[u]:
            return

        # remove the edge
        self.adj_list[v].remove(u)
        self.adj_list[u].remove(v)


    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges

        :param v: vertex to be removed.
        """
        # vertex does not exist
        if v not in self.adj_list:
            return

        # remove vertex and incident edges
        while self.adj_list[v]:
            self.remove_edge(v, self.adj_list[v][0])

        del self.adj_list[v]


    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order)
        """
        vertex_list = [v for v in self.adj_list]
        return vertex_list
       

    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order)
        """
        edge_list = []

        # iterate over every key-value pair in the graph
        for vertex, adjacent_vertices in self.adj_list.items():
            # for each vertex, copy the edge and add it to the output
            # list if it is not already in the list
            for adjacent_vertex in adjacent_vertices:
                if vertex < adjacent_vertex:
                    temp_edge = (vertex, adjacent_vertex)
                else:
                    temp_edge = (adjacent_vertex, vertex)

                if temp_edge not in edge_list:
                    edge_list.append(temp_edge)

        return edge_list

        

    def is_valid_path(self, path: []) -> bool:
        """
        Return true if provided path is valid, False otherwise
        """
        # loop over provided list
        # if curr index has curr_index + 1 in an edge, proceed

        if not path:
            return True

        # need to handle cases where path is one vertex
        if len(path) == 1:
            if path[0] in self.adj_list:
                return True
            else:
                return False

        i = 0

        while i < len(path) - 1:
            curr_vertex = path[i]
            next_vertex = path[i+1]

            if curr_vertex not in self.adj_list or next_vertex not in self.adj_list:
                return False

            if next_vertex not in self.adj_list[curr_vertex]:
                return False

            i += 1

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        """
        visited_vertices = []

        # start vertex not in graph
        if v_start not in self.adj_list:
            return visited_vertices

        # v_end not in graph
        if v_end is not None and v_end not in self.adj_list:
            v_end = None

        # use a stack for DFS!
        # list will support stack-like behaviors
        travel_stack = [v_start]

        while travel_stack:
            curr_vertex = travel_stack.pop()

            if curr_vertex not in visited_vertices:
                visited_vertices.append(curr_vertex)

            # if we're at the end break, the loop prematurely
            if curr_vertex == v_end:
                break

            # get the direct successors
            curr_direct_successors = []
            for vertex in self.adj_list[curr_vertex]:
                if vertex not in visited_vertices:
                    curr_direct_successors.append(vertex)

            # we are going to 'pop', so reverse sort makes us pop in ascending order
            curr_direct_successors.sort(reverse=True)
            travel_stack.extend(curr_direct_successors)

        return visited_vertices


    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in alphabetical order
        """
        visited_vertices = []

        # start vertex not in graph
        if v_start not in self.adj_list:
            return visited_vertices

        # v_end not in graph
        if v_end is not None and v_end not in self.adj_list:
            v_end = None

        # use a queue for BFS!
        travel_deque = deque(v_start)

        while travel_deque:
            curr_vertex = travel_deque.popleft()

            if curr_vertex not in visited_vertices:
                visited_vertices.append(curr_vertex)

                # if we're at the end break, the loop prematurely
                if curr_vertex == v_end:
                    break

                # get the direct successors
                curr_direct_successors = []
                for vertex in self.adj_list[curr_vertex]:
                    if vertex not in visited_vertices:
                        curr_direct_successors.append(vertex)

                curr_direct_successors.sort()
                travel_deque.extend(curr_direct_successors)

        return visited_vertices
        

    def count_connected_components(self):
        """
        Return number of connected components in the graph
        """
        # probably involves iterating over every vertex
        # A component of an undirected graph is an induced subgraph in which any two vertices are
        # connected to each other by paths, and which is connected to no additional vertices
        # in the rest of the graph
        # Vertex with no edges

        # maybe try a list of lists. Do a path for every vertex
        # unique only, and then count those

        component_list = []
        for vertex in self.adj_list:
            curr_dfs = self.dfs(vertex)
            curr_dfs.sort()
            if curr_dfs not in component_list:
                component_list.append(curr_dfs)

        return len(component_list)
      

    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # begin by iterating over vertices

        for v_start in self.adj_list:
            # track vertices and parents
            vertex_parent_dict = {}
            for v in self.adj_list:
                vertex_parent_dict[v] = None

            visited_vertices = []

            # use a stack for DFS!
            # list will support stack-like behaviors
            travel_stack = [v_start]

            while travel_stack:
                curr_vertex = travel_stack.pop()

                if curr_vertex not in visited_vertices:
                    visited_vertices.append(curr_vertex)

                # get the direct successors
                curr_direct_successors = []
                for vertex in self.adj_list[curr_vertex]:
                    if vertex not in visited_vertices:
                        curr_direct_successors.append(vertex)
                        vertex_parent_dict[vertex] = curr_vertex
                    elif vertex != vertex_parent_dict[curr_vertex]:
                        return True

                # we are going to 'pop', so reverse sort makes us pop in ascending order
                curr_direct_successors.sort(reverse=True)
                travel_stack.extend(curr_direct_successors)

        return False


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)


    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    print(g.adj_list['D'])
    g.remove_vertex('D')
    print(g)


    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())

    # print('\ncustom has_cycle() example 1')
    # edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    # g = UndirectedGraph(edges)
    # test_cases = (
    #     'add QH', 'remove FG', 'remove GQ', 'remove HQ',
    #     'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
    #     'remove BC')
    # for case in test_cases:
    #     command, edge = case.split()
    #     u, v = edge
    #     g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
    # print(g.has_cycle())

    # print('\ncustom has_cycle() example 2')
    # edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    # g = UndirectedGraph(edges)
    # test_cases = (
    #     'add QH', 'remove FG', 'remove GQ', 'remove HQ',
    #     'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
    #     'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ')
    # for case in test_cases:
    #     command, edge = case.split()
    #     u, v = edge
    #     g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
    # print(g.has_cycle())


#  Per Wikipedia, regarding back edges:
#  All the back edges which DFS skips over are part of cycles.[5]
#  In an undirected graph, the edge to the parent of a node should not be counted as a back edge, but finding any other already visited vertex will indicate a back edge