from types import SimpleNamespace

from graph.meta_func.core_func import CoreFunc


class Edge:
    def __init__(self, index):
        self.node1 = SimpleNamespace(caption=f'left-{index}')
        self.node2 = SimpleNamespace(caption=f'right-{index}')
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation


def test_edge_proposals_are_bounded_and_preserve_order():
    core = CoreFunc()
    core.prompt_edge_proposal = 'pairs:'
    prompts = []

    def respond(batch_prompts):
        prompts.extend(batch_prompts)
        return [
            '\n'.join(f'relation-{offset + index}' for index in range(prompt.count('(')))
            for offset, prompt in zip((0, 16, 32), batch_prompts)
        ]

    core.get_llm_responses = respond
    edges = [Edge(index) for index in range(35)]

    core.propose_edge_relations(edges)

    assert [prompt.count('(') for prompt in prompts] == [16, 16, 3]
    assert [edge.relation for edge in edges] == [f'relation-{index}' for index in range(35)]


def test_malformed_batch_does_not_assign_partial_relations():
    core = CoreFunc()
    core.prompt_edge_proposal = 'pairs:'
    core.get_llm_responses = lambda prompts: ['next to']
    edges = [Edge(index) for index in range(2)]

    core.propose_edge_relations(edges)

    assert [edge.relation for edge in edges] == [None, None]
