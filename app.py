"""
Game Environment State Machine & Path Inspector
A Streamlit Web Application for Formal Automata Theory & NPC Behavior Management.

Built for academic review and video game state inspection:
- Deterministic Finite Automaton (DFA) for legality verification.
- Mealy Finite State Transducer (FST) for event-driven output/action dispatch.
- Interactive live graph visualization with real-time state highlighting.
- Batch sequence validator with step-by-step diagnostic traces.
- Custom additive state/transition editor with reachability safety checks.
"""

import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.automata import DFA, FST, IllegalTransitionError
from src.state_machine import (
    NPCStateMachine,
    build_default_fst,
    DEFAULT_STATES,
    DEFAULT_ALPHABET,
    DEFAULT_OUTPUT_ALPHABET,
)
from src.visualizer import render_automaton_graph

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Game Environment State Machine & Path Inspector",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS Styling
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


# Initialize Session State Persistence
def init_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "state_machine" not in st.session_state:
        st.session_state.state_machine = NPCStateMachine()

    if "last_action_feedback" not in st.session_state:
        st.session_state.last_action_feedback = {
            "status": "INITIAL",
            "message": "NPC spawned in 'Idle' state. Ready for event triggers.",
            "output": "spawn_npc_idle",
            "event": "INITIALIZE",
            "timestamp": "-",
        }

    if "batch_input_text" not in st.session_state:
        st.session_state.batch_input_text = "StartPatrol, PlayerSpotted, PlayerAdjacent, HealthZero"

    if "custom_editor_warning" not in st.session_state:
        st.session_state.custom_editor_warning = None


init_session_state()
fsm: NPCStateMachine = st.session_state.state_machine


# Helper Functions for UI Badges & Statuses
def get_state_badge_html(state_name: str) -> str:
    """Generate HTML badge for a state name."""
    css_class_map = {
        "Idle": "state-badge-idle",
        "Patrol": "state-badge-patrol",
        "Chase": "state-badge-chase",
        "Attack": "state-badge-attack",
        "Dead": "state-badge-dead",
    }
    css_class = css_class_map.get(state_name, "state-badge-custom")
    icon_map = {
        "Idle": "🛡️",
        "Patrol": "🚶",
        "Chase": "🏃",
        "Attack": "⚔️",
        "Dead": "☠️",
    }
    icon = icon_map.get(state_name, "💠")
    return f'<span class="state-badge {css_class}">{icon} {state_name}</span>'


def get_output_badge_html(output_name: str) -> str:
    """Generate HTML badge for an output action."""
    if output_name == "NONE" or not output_name:
        return '<span class="output-pill" style="color:#94a3b8; border-color:#475569;">NONE</span>'
    return f'<span class="output-pill">🔊 {output_name}</span>'


# ==============================================================================
# SIDEBAR: Custom State & Transition Editor + Automaton Health Check
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Automaton Configuration")
    st.caption("Manage formal states, alphabet, and Mealy transitions.")

    # Automaton Cardinality Statistics
    q_len = len(fsm.fst.states)
    sigma_len = len(fsm.fst.alphabet)
    gamma_len = len(fsm.fst.output_alphabet)
    delta_len = len(fsm.fst.transition_table)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("States |Q|", q_len)
        st.metric("Alphabet |Σ|", sigma_len)
    with col_s2:
        st.metric("Outputs |Γ|", gamma_len)
        st.metric("Transitions |δ|", delta_len)

    st.divider()

    # Reachability & Structural Health Analysis
    reachability = fsm.analyze_reachability()
    if not reachability["is_fully_connected"]:
        unreach = ", ".join(sorted(reachability["unreachable_states"]))
        st.warning(f"⚠️ **Orphaned / Unreachable States Detected:** `{unreach}`")
    if reachability["dead_end_states"]:
        dead_ends = ", ".join(sorted(reachability["dead_end_states"]))
        st.warning(f"⚠️ **Non-terminal Dead-End States:** `{dead_ends}` (states with no valid outgoing transitions)")

    # Accordion 1: Add New State
    with st.expander("➕ Add New State to Q", expanded=False):
        new_state_name = st.text_input("State Identifier", placeholder="e.g. Stunned / Fleeing").strip()
        is_terminal = st.checkbox("Mark as Absorbing / Terminal State", value=False)

        if st.button("Add State", use_container_width=True):
            if not new_state_name:
                st.error("State name cannot be empty.")
            elif new_state_name in fsm.fst.states:
                st.info(f"State '{new_state_name}' is already in Q.")
            else:
                fsm.fst.states.add(new_state_name)
                fsm.fst.accept_states.add(new_state_name)
                if is_terminal:
                    fsm.fst.terminal_states.add(new_state_name)
                st.success(f"State '{new_state_name}' added to Q.")
                st.rerun()

    # Accordion 2: Add Custom Transition
    with st.expander("🔀 Add / Edit Transition δ(q, σ)", expanded=False):
        all_states = sorted(list(fsm.fst.states))
        from_st = st.selectbox("Current State (q)", all_states, key="editor_from_st")

        if from_st in fsm.fst.terminal_states:
            st.error(f"Cannot add outgoing transitions from terminal state '{from_st}'.")
        else:
            event_type = st.radio("Event Selection", ["Existing Alphabet", "Custom New Event"], horizontal=True)
            if event_type == "Existing Alphabet":
                event_choice = st.selectbox("Event (σ)", sorted(list(fsm.fst.alphabet)))
            else:
                event_choice = st.text_input("New Event Name (σ)", placeholder="e.g. FlashbangHit").strip()

            to_st = st.selectbox("Next State (q')", all_states, key="editor_to_st")

            output_type = st.radio("Output Selection", ["Existing Output", "Custom New Output"], horizontal=True)
            if output_type == "Existing Output":
                output_choice = st.selectbox("Output Action (γ)", sorted(list(fsm.fst.output_alphabet)))
            else:
                output_choice = st.text_input("New Output Name (γ)", placeholder="e.g. play_stun_sound").strip()

            if st.button("Save Transition", use_container_width=True):
                if not event_choice:
                    st.error("Event name cannot be empty.")
                elif not output_choice:
                    st.error("Output action cannot be empty.")
                else:
                    fsm.fst.add_transition(
                        from_state=from_st,
                        event=event_choice,
                        to_state=to_st,
                        output=output_choice,
                    )
                    st.success(f"Added: ({from_st}, {event_choice}) ➔ ({to_st}, {output_choice})")
                    st.rerun()

    st.divider()

    # Preset Restoration & Configuration Management
    st.markdown("#### 🔄 Preset Controls")
    if st.button("Reset to Default Reference Preset", use_container_width=True, type="secondary"):
        st.session_state.state_machine = NPCStateMachine(build_default_fst())
        st.session_state.last_action_feedback = {
            "status": "INITIAL",
            "message": "Automaton restored to reference preset.",
            "output": "spawn_npc_idle",
            "event": "RESET_PRESET",
            "timestamp": "-",
        }
        st.success("Restored reference automaton preset!")
        st.rerun()

    # Export Configuration as JSON
    config_dict = {
        "states": sorted(list(fsm.fst.states)),
        "alphabet": sorted(list(fsm.fst.alphabet)),
        "output_alphabet": sorted(list(fsm.fst.output_alphabet)),
        "initial_state": fsm.fst.initial_state,
        "terminal_states": sorted(list(fsm.fst.terminal_states)),
        "accept_states": sorted(list(fsm.fst.accept_states)),
        "transitions": [
            {
                "from_state": k[0],
                "event": k[1],
                "to_state": v,
                "output": fsm.fst.output_table.get(k, "none"),
            }
            for k, v in fsm.fst.transition_table.items()
        ],
    }
    json_str = json.dumps(config_dict, indent=2)
    st.download_button(
        label="💾 Export FSM Config (JSON)",
        data=json_str,
        file_name="npc_state_machine_config.json",
        mime="application/json",
        use_container_width=True,
    )


# ==============================================================================
# MAIN PAGE HEADER & HUD METRIC STRIP
# ==============================================================================
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">🎮 Game Environment State Machine & Path Inspector</h1>
        <p class="hero-subtitle">
            Formal Automata Theory Dashboard for Video Game NPC Behavior Modeling:
            Deterministic Finite Automata (DFA) Legality Engine & Mealy Finite State Transducer (FST).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# HUD Top Metric Cards
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns([1.3, 1.3, 1.0, 1.4])

with metric_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Current NPC State (q)</div>
            <div class="metric-value">{get_state_badge_html(fsm.current_state)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col2:
    last_out = st.session_state.last_action_feedback.get("output", "NONE")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Transducer Output (γ)</div>
            <div class="metric-value">{get_output_badge_html(last_out)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col3:
    step_count = len(fsm.history) - 1
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Step Traces</div>
            <div class="metric-value" style="color: #60a5fa;">{max(0, step_count)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col4:
    if fsm.is_dead:
        status_html = '<span style="color:#f87171; font-weight:700;">☠️ TERMINATED (Dead)</span>'
    elif fsm.current_state == "Attack":
        status_html = '<span style="color:#fb7185; font-weight:700;">⚔️ ENGAGED (Combat)</span>'
    elif fsm.current_state == "Chase":
        status_html = '<span style="color:#fbbf24; font-weight:700;">🏃 PURSUING (Alert)</span>'
    elif fsm.current_state == "Patrol":
        status_html = '<span style="color:#34d399; font-weight:700;">🚶 PATROLLING (Normal)</span>'
    else:
        status_html = '<span style="color:#93c5fd; font-weight:700;">🛡️ DORMANT (Idle)</span>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Behavioral Condition</div>
            <div class="metric-value" style="font-size:1.1rem; padding-top:0.35rem;">{status_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# TABBED INTERFACE (Tab 1, Tab 2, Tab 3)
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Tab 1: Live Interactive Simulator",
    "🧪 Tab 2: Batch Sequence Validator",
    "📚 Tab 3: Formal Theory & Documentation",
])


# ==============================================================================
# TAB 1: LIVE INTERACTIVE SIMULATOR
# ==============================================================================
with tab1:
    sim_left, sim_right = st.columns([1.1, 1.4], gap="medium")

    with sim_left:
        st.markdown("### 🕹️ Action & Event Dispatcher")
        st.caption("Trigger an input event symbol $σ \\in \\Sigma$ to execute transition $\\delta(q, \\sigma) \\to (q', \\gamma)$.")

        valid_events = fsm.get_valid_events()
        all_alphabet = sorted(list(fsm.fst.alphabet))

        # Mode toggle: valid only vs all alphabet
        display_mode = st.radio(
            "Event Display Filter:",
            ["Valid Outgoing Events Only", "Show All Alphabet Events (Test Illegal Rejection)"],
            horizontal=True,
        )

        st.markdown("---")

        if fsm.is_dead:
            st.error(
                "☠️ **NPC is in the Absorbing Terminal State ('Dead').**\n\n"
                "By definition of an absorbing state, all outgoing transitions are prohibited. "
                "Any incoming event will be rejected by the DFA engine. Click **Reset Simulation** below to re-spawn."
            )

        # Render Event Trigger Buttons
        events_to_display = valid_events if display_mode == "Valid Outgoing Events Only" else all_alphabet

        if not events_to_display and display_mode == "Valid Outgoing Events Only" and not fsm.is_dead:
            st.warning("No outgoing transitions defined from this state.")

        # Grid of event buttons
        for event_name in events_to_display:
            is_valid = fsm.fst.is_valid_transition(fsm.current_state, event_name)
            output_hint = fsm.fst.get_transition_output(fsm.current_state, event_name)

            btn_label = f"▶ {event_name}"
            if is_valid and output_hint:
                btn_help = f"Valid Transition ➔ Emits '{output_hint}'"
            elif fsm.is_dead:
                btn_help = "Illegal: NPC is Dead (absorbing state)."
            else:
                btn_help = f"Illegal: Undefined transition δ('{fsm.current_state}', '{event_name}')."

            btn_type = "primary" if is_valid else "secondary"

            if st.button(
                btn_label,
                key=f"btn_event_{event_name}",
                help=btn_help,
                type=btn_type,
                use_container_width=True,
            ):
                success, output_or_err, new_state = fsm.trigger(event_name)
                timestamp = fsm.history[-1]["timestamp"] if fsm.history else "-"
                st.session_state.last_action_feedback = {
                    "status": "SUCCESS" if success else "REJECTED",
                    "message": (
                        f"Transition accepted: ({fsm.history[-1]['from_state']}, {event_name}) ➔ {new_state}"
                        if success
                        else output_or_err
                    ),
                    "output": output_or_err if success else "NONE",
                    "event": event_name,
                    "timestamp": timestamp,
                }
                st.rerun()

        st.markdown("---")

        # Response Banner / Diagnostic Card
        last_feedback = st.session_state.last_action_feedback
        if last_feedback["status"] == "SUCCESS":
            st.success(
                f"✅ **Event `{last_feedback['event']}` Executed!**\n\n"
                f"- **Transducer Output:** `{last_feedback['output']}`\n"
                f"- **Result:** {last_feedback['message']}"
            )
        elif last_feedback["status"] == "REJECTED":
            st.error(
                f"🚫 **Transition REJECTED (Illegal Input)!**\n\n"
                f"**Engine Diagnostic:** {last_feedback['message']}\n\n"
                f"*Note: NPC remained in state `{fsm.current_state}` without state deviation.*"
            )
        else:
            st.info(f"ℹ️ {last_feedback['message']}")

        # Quick Simulator Controls Toolbar
        st.markdown("#### 🛠️ Simulation Controls")
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("↺ Reset to Idle", use_container_width=True):
                fsm.reset()
                st.session_state.last_action_feedback = {
                    "status": "INITIAL",
                    "message": "NPC state reset to initial state 'Idle'.",
                    "output": "spawn_npc_idle",
                    "event": "RESET",
                    "timestamp": "-",
                }
                st.rerun()
        with ctrl_col2:
            if st.button("🗑 Clear Log", use_container_width=True):
                fsm.history.clear()
                fsm._record_initial_state()
                st.rerun()

    # Right Column: Live Graphviz State Machine Diagram
    with sim_right:
        st.markdown("### 🗺️ Live Directed State Graph")
        st.caption(
            "Visual Mealy FST: Active state highlighted in **green**, "
            "terminal Dead state in **red**, transitions labeled with `event / output`."
        )

        try:
            dot_graph = render_automaton_graph(fsm.fst, current_state=fsm.current_state)
            st.graphviz_chart(dot_graph.source, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Graphviz diagram: {e}")

        st.caption("💡 The start indicator arrow points to initial state $q_0 = \\text{Idle}$.")

    st.divider()

    # Bottom Section: Audit History Log Table
    st.markdown("### 📋 Transition Audit & Execution Log")
    st.caption("Complete chronological record of all valid and rejected transitions.")

    history_df = fsm.get_history_df()

    if not history_df.empty:
        filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1, 1])
        with filter_col1:
            status_filter = st.selectbox(
                "Filter Log Entries by Status:",
                ["All Records", "SUCCESS Only", "REJECTED Only"],
                index=0,
            )

        filtered_df = history_df.copy()
        if status_filter == "SUCCESS Only":
            filtered_df = filtered_df[filtered_df["Status"] == "SUCCESS"]
        elif status_filter == "REJECTED Only":
            filtered_df = filtered_df[filtered_df["Status"] == "REJECTED"]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Step": st.column_config.NumberColumn("Step #", width="small"),
                "Timestamp": st.column_config.TextColumn("Time", width="small"),
                "From State": st.column_config.TextColumn("From (q)", width="medium"),
                "Event": st.column_config.TextColumn("Event (σ)", width="medium"),
                "To State": st.column_config.TextColumn("To (q')", width="medium"),
                "Output Emitted": st.column_config.TextColumn("Output (γ)", width="medium"),
                "Status": st.column_config.TextColumn("Verdict", width="small"),
                "Details": st.column_config.TextColumn("Diagnostic Details", width="large"),
            },
        )

        # Export Buttons
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv_data,
            file_name="npc_state_machine_history.csv",
            mime="text/csv",
        )


# ==============================================================================
# TAB 2: BATCH SEQUENCE VALIDATOR
# ==============================================================================
with tab2:
    st.markdown("### 🧪 Batch Sequence Validator & Path Inspector")
    st.caption(
        "Validate arbitrary event sequences $w = \\sigma_1 \\sigma_2 \\dots \\sigma_n$ against the formal DFA "
        "from a clean initial state. Inspect step-by-step state transitions and Mealy action outputs."
    )

    # Preset Sequence Buttons
    st.markdown("#### ⚡ Quick-Load Test Presets")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        if st.button("🎯 Valid Combat Run", use_container_width=True):
            st.session_state.batch_input_text = "StartPatrol, PlayerSpotted, PlayerAdjacent, HealthZero"
            st.rerun()

    with p_col2:
        if st.button("⚠️ Illegal Direct Attack", use_container_width=True):
            st.session_state.batch_input_text = "StartPatrol, PlayerAdjacent"
            st.rerun()

    with p_col3:
        if st.button("☠️ Post-Mortem Event", use_container_width=True):
            st.session_state.batch_input_text = "HealthZero, StartPatrol"
            st.rerun()

    with p_col4:
        if st.button("⚔️ Extended Skirmish Loop", use_container_width=True):
            st.session_state.batch_input_text = (
                "StartPatrol, TakeDamage, PlayerAdjacent, PlayerOutOfRange, PlayerAdjacent, HealthZero"
            )
            st.rerun()

    # Input text area
    batch_input = st.text_area(
        "Input Event Sequence (comma-separated or newline-separated):",
        value=st.session_state.batch_input_text,
        height=90,
        help="Enter event symbols exactly matching the defined alphabet Sigma.",
    )

    validate_btn = st.button("🚀 Run Batch Validation", type="primary", use_container_width=True)

    if batch_input and (validate_btn or "batch_input_text" in st.session_state):
        # Parse and sanitize sequence
        raw_events = [item.strip() for item in batch_input.replace("\n", ",").split(",") if item.strip()]

        if not raw_events:
            st.warning("Please provide at least one event in the sequence.")
        else:
            # Perform formal batch validation with outputs
            fresh_fst = build_default_fst()
            # If user customized states in session state, use session fst transitions
            fresh_fst.states = set(fsm.fst.states)
            fresh_fst.alphabet = set(fsm.fst.alphabet)
            fresh_fst.output_alphabet = set(fsm.fst.output_alphabet)
            fresh_fst.transition_table = dict(fsm.fst.transition_table)
            fresh_fst.output_table = dict(fsm.fst.output_table)
            fresh_fst.terminal_states = set(fsm.fst.terminal_states)
            fresh_fst.accept_states = set(fsm.fst.accept_states)

            is_accepted, trace_steps = fresh_fst.validate_with_outputs(raw_events)

            st.markdown("---")

            # Verdict Banner
            if is_accepted:
                st.markdown(
                    """
                    <div style="background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
                        <h3 style="color: #6ee7b7; margin:0;">✅ ACCEPTED: Formal Sequence is Valid (w ∈ L(M))</h3>
                        <p style="color: #d1fae5; margin: 0.3rem 0 0 0;">
                            All transitions were deterministically resolved and the automaton concluded in a valid accept state.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
                        <h3 style="color: #fca5a5; margin:0;">🚫 REJECTED: Sequence Violates Automaton Grammar (w ∉ L(M))</h3>
                        <p style="color: #fee2e2; margin: 0.3rem 0 0 0;">
                            The sequence encountered an undefined transition or an illegal event from an absorbing state.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Trace Summary Metrics
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                st.metric("Total Events Provided", len(raw_events))
            with t_col2:
                valid_count = sum(1 for step in trace_steps if step["valid"])
                st.metric("Valid Transitions", f"{valid_count} / {len(raw_events)}")
            with t_col3:
                terminal_state_reached = any(step.get("is_terminal", False) for step in trace_steps if step["valid"])
                st.metric("Terminal State Reached?", "Yes (Dead)" if terminal_state_reached else "No")
            with t_col4:
                final_state = trace_steps[-1]["to_state"] if (trace_steps and trace_steps[-1]["valid"]) else trace_steps[-1]["from_state"] if trace_steps else "Idle"
                st.metric("Final Machine State", final_state)

            # Step-by-Step Diagnostic Table
            st.markdown("#### 🔬 Step-by-Step Diagnostic Trace")
            table_rows = []
            for step in trace_steps:
                table_rows.append({
                    "Step #": step["step"],
                    "Event Symbol (σ)": step["event"],
                    "State Before (q)": step["from_state"],
                    "Transition Status": "✅ ACCEPTED" if step["valid"] else "❌ REJECTED",
                    "State After (q')": step["to_state"] if step["to_state"] else f"{step['from_state']} (UNMOVED)",
                    "Transducer Output (γ)": step.get("output", "NONE"),
                    "Diagnostic Note": step["error"] if step["error"] else "Valid transition executed.",
                })

            trace_df = pd.DataFrame(table_rows)
            st.dataframe(trace_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: FORMAL THEORY & DOCUMENTATION
# ==============================================================================
with tab3:
    st.markdown("### 📚 Formal Automata Theory & Project Documentation")
    st.caption("Academic foundations, mathematical definitions, and architectural design rationale for college presentation.")

    doc_col1, doc_col2 = st.columns([1, 1], gap="large")

    with doc_col1:
        st.markdown(
            """
            <div class="theory-card">
                <h4>1. Deterministic Finite Automaton (DFA) Definition</h4>
                <p>Formally, the DFA governing the NPC state legality is defined as the 5-tuple:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"M_{\text{DFA}} = (Q, \Sigma, \delta, q_0, F)")

        st.markdown(
            r"""
            - **$Q$ (States):** Finite set of discrete NPC behavioral modes:
              $$\{ \text{Idle}, \text{Patrol}, \text{Chase}, \text{Attack}, \text{Dead} \}$$
            - **$\Sigma$ (Input Alphabet):** Discrete external stimuli/gameplay events:
              $$\{ \text{StartPatrol}, \text{PlayerSpotted}, \text{PlayerLost}, \text{PlayerAdjacent}, \text{PlayerOutOfRange}, \text{TakeDamage}, \text{HealthZero} \}$$
            - **$\delta$ (Transition Function):** Partial mapping $\delta: Q \times \Sigma \to Q$.
            - **$q_0$ (Start State):** Initial spawned state $q_0 = \text{Idle}$.
            - **$F$ (Accept States):** Valid behavioral states $F = Q$.
            """
        )

        st.markdown(
            """
            <div class="theory-card">
                <h4>2. Mealy Machine Finite State Transducer (FST)</h4>
                <p>To produce reactive game actions (audio triggers, animations), the DFA is extended to a 6-tuple Mealy transducer:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"M_{\text{FST}} = (Q, \Sigma, \Gamma, \delta, \omega, q_0)")

        st.markdown(
            r"""
            - **$\Gamma$ (Output Alphabet):** Discrete action dispatches emitted during transitions:
              $$\{ \text{begin\_patrol\_route}, \text{play\_alert\_sound}, \text{resume\_patrol}, \text{play\_attack\_animation}, \text{resume\_chase}, \text{play\_death\_animation} \}$$
            - **$\omega$ (Output Function):** $\omega: Q \times \Sigma \to \Gamma$, bound directly to the state transition:
              $$\delta(q, \sigma) \to (q', \gamma)$$
            """
        )

    with doc_col2:
        st.markdown(
            """
            <div class="theory-card">
                <h4>3. Architectural Rationale: Mealy vs. Moore Machines</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            In video game architecture, selecting between **Mealy** and **Moore** machines has critical performance and semantic implications:

            | Characteristic | Moore Machine | Mealy Machine (Used in this Project) |
            |---|---|---|
            | **Output Binding** | Outputs depend *only* on state: $\\lambda(q)$ | Outputs depend on *transition*: $\\omega(q, \\sigma)$ |
            | **State Explosion** | Requires extra states to emit different cues for different entry triggers | Handles diverse triggers entering the same state cleanly |
            | **Event Reactivity** | Delayed until state update cycle | Immediate dispatch on transition invocation |
            | **Game Loop Fit** | Requires polling state entrance/exit | Ideal for event-driven audio/VFX trigger queues |

            **Concrete Example:**
            Both `PlayerSpotted` and `TakeDamage` transition the NPC from `Patrol` into `Chase`.
            In a Mealy machine, both transitions cleanly emit `play_alert_sound` without needing duplicated intermediate states.
            """
        )

        st.markdown(
            """
            <div class="theory-card">
                <h4>4. Absorbing / Terminal State Properties</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - **Absorbing Definition:** A state $q_{\\text{term}} \\in Q$ is terminal/absorbing if:
              $$\\forall \\sigma \\in \\Sigma, \\quad \\delta(q_{\\text{term}}, \\sigma) = \\emptyset \\quad \\text{(Undefined)}$$
            - In this automaton, **`Dead`** is an absorbing state.
            - Once entered, **any event** received by the NPC is explicitly rejected, ensuring dead entities cannot resume patrol or attack.
            """
        )

    st.divider()

    # Complete State Transition Matrix Table
    st.markdown("### 📊 Complete State Transition & Output Matrix $\\delta(q, \\sigma) \\to (q', \\gamma)$")
    
    matrix_data = [
        {"Current State (q)": "Idle", "Input Event (σ)": "StartPatrol", "Next State (q')": "Patrol", "Output Emitted (γ)": "begin_patrol_route", "Type": "Operational"},
        {"Current State (q)": "Patrol", "Input Event (σ)": "PlayerSpotted", "Next State (q')": "Chase", "Output Emitted (γ)": "play_alert_sound", "Type": "Operational"},
        {"Current State (q)": "Patrol", "Input Event (σ)": "TakeDamage", "Next State (q')": "Chase", "Output Emitted (γ)": "play_alert_sound", "Type": "Operational"},
        {"Current State (q)": "Chase", "Input Event (σ)": "PlayerLost", "Next State (q')": "Patrol", "Output Emitted (γ)": "resume_patrol", "Type": "Operational"},
        {"Current State (q)": "Chase", "Input Event (σ)": "PlayerAdjacent", "Next State (q')": "Attack", "Output Emitted (γ)": "play_attack_animation", "Type": "Operational"},
        {"Current State (q)": "Attack", "Input Event (σ)": "PlayerOutOfRange", "Next State (q')": "Chase", "Output Emitted (γ)": "resume_chase", "Type": "Operational"},
        {"Current State (q)": "Attack", "Input Event (σ)": "PlayerLost", "Next State (q')": "Patrol", "Output Emitted (γ)": "resume_patrol", "Type": "Operational"},
        {"Current State (q)": "Idle", "Input Event (σ)": "HealthZero", "Next State (q')": "Dead", "Output Emitted (γ)": "play_death_animation", "Type": "Terminal (Absorbing)"},
        {"Current State (q)": "Patrol", "Input Event (σ)": "HealthZero", "Next State (q')": "Dead", "Output Emitted (γ)": "play_death_animation", "Type": "Terminal (Absorbing)"},
        {"Current State (q)": "Chase", "Input Event (σ)": "HealthZero", "Next State (q')": "Dead", "Output Emitted (γ)": "play_death_animation", "Type": "Terminal (Absorbing)"},
        {"Current State (q)": "Attack", "Input Event (σ)": "HealthZero", "Next State (q')": "Dead", "Output Emitted (γ)": "play_death_animation", "Type": "Terminal (Absorbing)"},
        {"Current State (q)": "Dead", "Input Event (σ)": "Any Event", "Next State (q')": "REJECTED (Undefined)", "Output Emitted (γ)": "NONE", "Type": "Illegal Transition"},
    ]

    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Game Environment State Machine & Path Inspector • College Project • Automata Theory & Game AI Simulation
    </div>
    """,
    unsafe_allow_html=True,
)
