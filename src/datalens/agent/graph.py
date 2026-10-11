"""Build the LangGraph workflow for DataLens.

The graph is deliberately explicit: every node and edge is visible here so
the control flow can be reasoned about without chasing decorators.
"""

from langgraph.graph import END, START, StateGraph

from datalens.agent.nodes import (
    coder_node,
    executor_node,
    fixer_node,
    planner_node,
    route_after_executor,
    route_after_planner,
    summarizer_node,
)
from datalens.agent.state import AgentState


def build_agent_graph():
    """Construct and compile the DataLens agent graph."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("executor", executor_node)
    graph.add_node("fixer", fixer_node)
    graph.add_node("summarizer", summarizer_node)

    # Linear path up to the executor
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "coder": "coder",
            "end": END,
        },
    )
    graph.add_edge("coder", "executor")

    # After executor: fixer loop, success path, or give-up
    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "summarizer": "summarizer",
            "fixer": "fixer",
            "give_up": END,
        },
    )

    # Fixer loops back to executor for another attempt
    graph.add_edge("fixer", "executor")

    # Summarizer ends the graph
    graph.add_edge("summarizer", END)

    return graph.compile()
