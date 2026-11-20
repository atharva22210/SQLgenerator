"""Select connecting paths, then remove cycles and unnecessary bridge leaves.

Read select_connections last. The small disjoint-set helper tracks connected
groups; Kruskal adds a connection only when it joins different groups.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

from rag.hybrid_graph.graph import GraphPath


class DisjointSets:
    """Track groups of table IDs; join returns False for an existing connection."""
    def __init__(self):
        self.parent = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            next_node = self.parent[node]
            self.parent[node] = root
            node = next_node
        return root

    def join(self, source: str, target: str) -> bool:
        left, right = self.find(source), self.find(target)
        if left == right:
            return False
        self.parent[max(left, right)] = min(left, right)
        return True


def edge_nodes(edge: dict) -> tuple[str, str]:
    """Return the endpoints while retaining declared FK direction in the record."""
    return edge["from_document_id"], edge["to_document_id"]


def minimum_forest(edge_ids: set[int], edges: list[dict], costs: dict[int, float]) -> set[int]:
    """Remove cycles from a path union using question-weighted Kruskal.

    edge_ids refers only to already selected permitted FK edges. Keep the
    cheapest edges that preserve its connected components, with stable ties.
    This optimizes within this union, not over every possible schema subgraph.
    """
    groups, retained = DisjointSets(), set()
    ordered = sorted(edge_ids, key=lambda i: (costs[i], tuple(sorted(edge_nodes(edges[i]))), i))
    for edge_id in ordered:
        if groups.join(*edge_nodes(edges[edge_id])):
            retained.add(edge_id)
    return retained


def prune_bridge_leaves(edge_ids: set[int], edges: list[dict], seeds: set[str]) -> set[int]:
    """Repeatedly remove degree-one non-seeds without disconnecting seed pairs.

    A bridge that ends a branch cannot connect two seeds. Removing it may expose
    another unnecessary leaf, so continue until only seed-supporting branches
    remain. Seeds are never removed, even if they are disconnected or leaves.
    """
    retained = set(edge_ids)
    adjacency = defaultdict(set)
    for edge_id in retained:
        for node in edge_nodes(edges[edge_id]):
            adjacency[node].add(edge_id)
    queue = deque(sorted(node for node in adjacency if node not in seeds and len(adjacency[node]) == 1))
    while queue:
        node = queue.popleft()
        if node in seeds or len(adjacency[node]) != 1:
            continue
        edge_id = next(iter(adjacency[node]))
        retained.remove(edge_id)
        for endpoint in edge_nodes(edges[edge_id]):
            adjacency[endpoint].remove(edge_id)
            if endpoint not in seeds and len(adjacency[endpoint]) == 1:
                queue.append(endpoint)
    return retained


@dataclass(frozen=True)
class ConnectionSelection:
    """Paths before cleanup, final FK edges, and diagnostics for explanations."""
    paths: tuple[GraphPath, ...]
    edge_ids: tuple[int, ...]
    skipped_paths: tuple[GraphPath, ...]
    redundant_paths: int
    pruned_document_ids: tuple[str, ...]
    seed_groups: tuple[tuple[str, ...], ...]


def select_connections(paths: list[GraphPath], edges: list[dict], costs: dict[int, float],
                       seeds: set[str], *, max_added_tables: int) -> ConnectionSelection:
    """Choose affordable paths joining separate groups, then prune their union.

    paths: One shortest path per reachable same-database seed pair.
    costs: Positive question-dependent costs for permitted FK edge IDs.
    seeds: Starting table IDs to preserve. max_added_tables: Whole-path budget.
    A selected path can pass through other seeds and connect several groups at
    once. Skip later paths whose endpoints are already connected. Under budgets
    or disconnected schemas this returns a forest, not a forced complete tree.
    """
    groups, included, chosen_edges = DisjointSets(), set(seeds), set()
    chosen, skipped, redundant = [], [], 0
    for path in sorted(paths, key=lambda p: (p.cost, p.nodes, p.edge_ids)):
        if groups.find(path.nodes[0]) == groups.find(path.nodes[-1]):
            redundant += 1
            continue
        additions = set(path.nodes) - included
        if len(included | additions) > len(seeds) + max_added_tables:
            skipped.append(path)
            continue
        chosen.append(path)
        included.update(additions)
        chosen_edges.update(path.edge_ids)
        for edge_id in path.edge_ids:
            groups.join(*edge_nodes(edges[edge_id]))

    retained = minimum_forest(chosen_edges, edges, costs)
    retained = prune_bridge_leaves(retained, edges, seeds)
    final_nodes, final_groups = set(seeds), DisjointSets()
    for edge_id in retained:
        source, target = edge_nodes(edges[edge_id])
        final_nodes.update((source, target))
        final_groups.join(source, target)
    components = defaultdict(list)
    for seed in sorted(seeds):
        components[final_groups.find(seed)].append(seed)
    return ConnectionSelection(tuple(chosen), tuple(sorted(retained)), tuple(skipped), redundant,
                               tuple(sorted(included - final_nodes)),
                               tuple(sorted(tuple(group) for group in components.values())))
