"""Small hand-constructed graphs verify connection selection and leaf pruning."""
import unittest

from rag.hybrid_graph.connections import minimum_forest, prune_bridge_leaves, select_connections
from rag.hybrid_graph.graph import GraphPath


def edge(source, target):
    return {"from_document_id": source, "to_document_id": target}


class ConnectionTests(unittest.TestCase):
    def test_selects_two_paths_instead_of_triangle_and_omits_unneeded_bridge(self):
        edges = [edge("a", "x"), edge("x", "b"), edge("b", "y"), edge("y", "c"),
                 edge("a", "z"), edge("z", "c")]
        costs = dict(enumerate([.5, .5, .6, .6, .9, .9]))
        paths = [GraphPath(("a", "x", "b"), (0, 1), 1.0),
                 GraphPath(("b", "y", "c"), (2, 3), 1.2),
                 GraphPath(("a", "z", "c"), (4, 5), 1.8)]
        result = select_connections(paths, edges, costs, {"a", "b", "c"}, max_added_tables=10)
        self.assertEqual(result.edge_ids, (0, 1, 2, 3))
        self.assertEqual(result.seed_groups, (("a", "b", "c"),))
        self.assertEqual(result.redundant_paths, 1)
        self.assertEqual(len(result.paths), 2)
        # Input path order must not change the selected structure.
        self.assertEqual(result, select_connections(list(reversed(paths)), edges, costs,
                                                    {"a", "b", "c"}, max_added_tables=10))

    def test_cycle_removal_then_recursive_non_seed_leaf_pruning(self):
        edges = [edge("a", "x"), edge("b", "x"), edge("b", "y"), edge("y", "x"),
                 edge("x", "c"), edge("y", "z")]
        costs = dict(enumerate([1, 1, 2, 1, 1, 1]))
        forest = minimum_forest(set(range(6)), edges, costs)
        self.assertNotIn(2, forest)  # Drop expensive b-y cycle edge.
        # z is removed, then y becomes another removable bridge leaf.
        pruned = prune_bridge_leaves(forest, edges, {"a", "b", "c"})
        self.assertEqual(pruned, {0, 1, 4})
        self.assertIn(5, prune_bridge_leaves(forest, edges, {"a", "b", "c", "z"}))

    def test_disconnected_and_budget_limited_groups_are_reported(self):
        edges = [edge("a", "x"), edge("x", "b")]
        path = GraphPath(("a", "x", "b"), (0, 1), 1)
        result = select_connections([path], edges, {0:.5, 1:.5}, {"a", "b", "c"}, max_added_tables=0)
        self.assertEqual(result.edge_ids, ())
        self.assertEqual(result.seed_groups, (("a",), ("b",), ("c",)))
        self.assertEqual(len(result.skipped_paths), 1)
        result = select_connections([path], edges, {0:.5, 1:.5}, {"a", "b", "c"}, max_added_tables=1)
        self.assertEqual(result.seed_groups, (("a", "b"), ("c",)))

    def test_path_through_third_seed_connects_it_without_another_path(self):
        edges = [edge("a", "b"), edge("b", "c")]
        paths = [GraphPath(("a", "b", "c"), (0, 1), 1), GraphPath(("b", "c"), (1,), 2)]
        result = select_connections(paths, edges, {0:.5, 1:.5}, {"a", "b", "c"}, max_added_tables=0)
        self.assertEqual(len(result.paths), 1)
        self.assertEqual(result.seed_groups, (("a", "b", "c"),))


if __name__ == "__main__":
    unittest.main()
