"""
Graphviz Directed Graph Visualizer for NPC Automaton State Machine.

This module provides visual rendering of the formal Mealy FST graph:
- Highlights the active current state in vibrant green.
- Renders the terminal/absorbing Dead state with a distinct red double-border/tint.
- Points a start node into the initial state.
- Formats transition edges as 'event / output' with distinct styling.
"""

from typing import Dict, List, Optional, Set, Tuple
import graphviz
from src.automata import FST, DFA


def render_automaton_graph(
    automaton: FST,
    current_state: Optional[str] = None,
    graph_attr: Optional[Dict[str, str]] = None,
) -> graphviz.Digraph:
    """
    Generate a styled Graphviz Digraph representing the automaton and highlighting the current state.

    Args:
        automaton (FST): The Mealy FST instance (or DFA).
        current_state (Optional[str]): Active state to highlight. If None, uses automaton.current_state.
        graph_attr (Optional[Dict[str, str]]): Additional graph attributes.

    Returns:
        graphviz.Digraph: Configured Graphviz Digraph object.
    """
    active_st = current_state if current_state is not None else automaton.current_state

    # Initialize Graphviz Directed Graph
    dot = graphviz.Digraph(
        name="NPC_Automaton_FST",
        comment="Formal Mealy Machine for NPC Game Behavior State Machine",
        format="svg",
    )

    # Global Graph Styling (Dark-mode optimized & presentation clean)
    dot.attr(
        rankdir="LR",
        bgcolor="transparent",
        dpi="150",
        pad="0.3",
        nodesep="0.65",
        ranksep="0.75",
        fontname="Inter, Segoe UI, sans-serif",
        fontsize="13",
        fontcolor="#f8fafc",
    )

    if graph_attr:
        dot.attr(**graph_attr)

    # Global Node Default Attributes
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Inter, Segoe UI, sans-serif",
        fontsize="12",
        fontcolor="#f8fafc",
        fillcolor="#1e293b",
        color="#475569",
        penwidth="1.8",
        margin="0.2,0.12",
    )

    # Global Edge Default Attributes
    dot.attr(
        "edge",
        fontname="Fira Code, Consolas, monospace",
        fontsize="9.5",
        fontcolor="#94a3b8",
        color="#64748b",
        arrowsize="0.85",
        penwidth="1.4",
    )

    # Virtual start node for initial state indicator
    start_node_id = "__start__"
    dot.node(
        start_node_id,
        label="START",
        shape="point",
        width="0.15",
        color="#3b82f6",
    )
    dot.edge(
        start_node_id,
        automaton.initial_state,
        label="",
        color="#3b82f6",
        penwidth="2.2",
        arrowsize="1.0",
    )

    # Render Nodes
    for state in sorted(list(automaton.states)):
        is_active = (state == active_st)
        is_terminal = (state in automaton.terminal_states)
        is_initial = (state == automaton.initial_state)

        # Build clean visual label
        badge_text = state
        if is_active:
            badge_text = f"▶ {state} (ACTIVE)"
        elif is_terminal:
            badge_text = f"☠ {state} (DEAD/TERMINAL)"

        # Style resolution
        if is_active and is_terminal:
            # Active in Dead state
            fillcolor = "#7f1d1d"
            border_color = "#ef4444"
            fontcolor = "#fee2e2"
            penwidth = "3.2"
            shape = "box"
            style = "rounded,filled,bold"
        elif is_active:
            # Active node - glowing green highlight
            fillcolor = "#065f46"
            border_color = "#10b981"
            fontcolor = "#ecfdf5"
            penwidth = "3.0"
            shape = "box"
            style = "rounded,filled,bold"
        elif is_terminal:
            # Terminal / Dead node - red accent
            fillcolor = "#450a0a"
            border_color = "#dc2626"
            fontcolor = "#fca5a5"
            penwidth = "2.2"
            shape = "box"
            style = "rounded,filled"
        elif is_initial:
            # Initial state node
            fillcolor = "#1e3a8a"
            border_color = "#3b82f6"
            fontcolor = "#dbeafe"
            penwidth = "2.0"
            shape = "box"
            style = "rounded,filled"
        else:
            # Standard operational node
            fillcolor = "#1e293b"
            border_color = "#475569"
            fontcolor = "#f8fafc"
            penwidth = "1.8"
            shape = "box"
            style = "rounded,filled"

        dot.node(
            state,
            label=badge_text,
            shape=shape,
            style=style,
            fillcolor=fillcolor,
            color=border_color,
            fontcolor=fontcolor,
            penwidth=penwidth,
        )

    # Group transitions between same (from_state, to_state) for clean edge labeling
    grouped_edges: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for (from_st, event), to_st in automaton.transition_table.items():
        output = automaton.output_table.get((from_st, event), "none") if isinstance(automaton, FST) else ""
        pair = (from_st, to_st)
        if pair not in grouped_edges:
            grouped_edges[pair] = []
        grouped_edges[pair].append((event, output))

    # Render Edges
    for (from_st, to_st), trans_list in grouped_edges.items():
        is_from_active = (from_st == active_st)

        # Build edge labels: "Event / output"
        labels: List[str] = []
        for ev, out in trans_list:
            if out:
                labels.append(f"{ev} / {out}")
            else:
                labels.append(f"{ev}")

        edge_label = " \\n ".join(labels)

        # Highlight outgoing edges from the active state
        if is_from_active:
            edge_color = "#38bdf8"
            label_color = "#7dd3fc"
            edge_width = "2.0"
        elif to_st in automaton.terminal_states:
            edge_color = "#f87171"
            label_color = "#fca5a5"
            edge_width = "1.6"
        else:
            edge_color = "#64748b"
            label_color = "#cbd5e1"
            edge_width = "1.3"

        dot.edge(
            from_st,
            to_st,
            label=f"  {edge_label}  ",
            color=edge_color,
            fontcolor=label_color,
            penwidth=edge_width,
        )

    return dot
