"""
Runtime Simulation Wrapper for NPC Game State Management.

This module provides the `NPCStateMachine` class which wraps the Mealy FST engine,
records execution histories, safely captures illegal transition exceptions,
and exports audit data for pandas tables and visualization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd

from src.automata import FST, IllegalTransitionError


# Default Reference Automaton Configuration Constants
DEFAULT_STATES: Set[str] = {"Idle", "Patrol", "Chase", "Attack", "Dead"}

DEFAULT_ALPHABET: Set[str] = {
    "StartPatrol",
    "PlayerSpotted",
    "PlayerLost",
    "PlayerAdjacent",
    "PlayerOutOfRange",
    "TakeDamage",
    "HealthZero",
}

DEFAULT_OUTPUT_ALPHABET: Set[str] = {
    "begin_patrol_route",
    "play_alert_sound",
    "resume_patrol",
    "play_attack_animation",
    "resume_chase",
    "play_death_animation",
}

DEFAULT_INITIAL_STATE: str = "Idle"
DEFAULT_TERMINAL_STATES: Set[str] = {"Dead"}
DEFAULT_ACCEPT_STATES: Set[str] = {"Idle", "Patrol", "Chase", "Attack", "Dead"}

DEFAULT_TRANSITIONS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Idle", "StartPatrol"): ("Patrol", "begin_patrol_route"),
    ("Patrol", "PlayerSpotted"): ("Chase", "play_alert_sound"),
    ("Patrol", "TakeDamage"): ("Chase", "play_alert_sound"),
    ("Chase", "PlayerLost"): ("Patrol", "resume_patrol"),
    ("Chase", "PlayerAdjacent"): ("Attack", "play_attack_animation"),
    ("Attack", "PlayerOutOfRange"): ("Chase", "resume_chase"),
    ("Attack", "PlayerLost"): ("Patrol", "resume_patrol"),
    ("Idle", "HealthZero"): ("Dead", "play_death_animation"),
    ("Patrol", "HealthZero"): ("Dead", "play_death_animation"),
    ("Chase", "HealthZero"): ("Dead", "play_death_animation"),
    ("Attack", "HealthZero"): ("Dead", "play_death_animation"),
}


def build_default_fst() -> FST:
    """Instantiate a new Mealy FST loaded with the reference automaton."""
    return FST(
        states=set(DEFAULT_STATES),
        alphabet=set(DEFAULT_ALPHABET),
        output_alphabet=set(DEFAULT_OUTPUT_ALPHABET),
        transitions=dict(DEFAULT_TRANSITIONS),
        initial_state=DEFAULT_INITIAL_STATE,
        accept_states=set(DEFAULT_ACCEPT_STATES),
        terminal_states=set(DEFAULT_TERMINAL_STATES),
    )


class NPCStateMachine:
    """
    High-level state machine runtime simulator for Non-Player Characters (NPCs).

    Features:
    - Encapsulates formal Mealy FST execution.
    - Non-crashing exception handling on illegal events (NPC stays in current state).
    - Immutable step-by-step history logging with timestamps.
    - Integration with Pandas DataFrame for dashboard rendering.
    """

    def __init__(self, fst: Optional[FST] = None) -> None:
        """Initialize the NPC simulation runtime with an FST instance or reference preset."""
        self.fst: FST = fst if fst is not None else build_default_fst()
        self.history: List[Dict[str, Any]] = []
        self._step_counter: int = 0
        self._record_initial_state()

    def _record_initial_state(self) -> None:
        """Record the baseline initialization state in the history log."""
        self.history.append({
            "step": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "from_state": "-",
            "event": "INITIALIZE",
            "to_state": self.fst.current_state,
            "output": "spawn_npc_idle",
            "status": "INITIAL",
            "message": f"NPC initialized in state '{self.fst.current_state}'.",
        })

    @property
    def current_state(self) -> str:
        """Get the current state of the NPC."""
        return self.fst.current_state

    @property
    def is_dead(self) -> bool:
        """Check if NPC is in absorbing/terminal Dead state."""
        return self.fst.current_state in self.fst.terminal_states

    def get_valid_events(self) -> List[str]:
        """Get list of valid outgoing events for the current state."""
        return self.fst.get_valid_events(self.fst.current_state)

    def trigger(self, event: str) -> Tuple[bool, str, str]:
        """
        Trigger an event in the NPC state machine.

        Args:
            event (str): Input event name.

        Returns:
            Tuple[bool, str, str]:
                - success: True if transition succeeded, False if illegal.
                - output_or_error: Emitted Mealy output string on success, or rejection error message.
                - active_state: Current state after event processing (unchanged on failure).
        """
        self._step_counter += 1
        from_state = self.fst.current_state
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        try:
            next_state, output = self.fst.step(event)
            record = {
                "step": self._step_counter,
                "timestamp": timestamp,
                "from_state": from_state,
                "event": event,
                "to_state": next_state,
                "output": output,
                "status": "SUCCESS",
                "message": f"Transitioned from '{from_state}' to '{next_state}' emitting '{output}'.",
            }
            self.history.append(record)
            return True, output, next_state

        except IllegalTransitionError as err:
            # Crucial requirement: NPC stays in current state on illegal input
            record = {
                "step": self._step_counter,
                "timestamp": timestamp,
                "from_state": from_state,
                "event": event,
                "to_state": from_state,
                "output": "NONE",
                "status": "REJECTED",
                "message": err.message,
            }
            self.history.append(record)
            return False, err.message, from_state

    def reset(self, initial_state: Optional[str] = None) -> None:
        """Reset the state machine back to start and refresh history."""
        self.fst.reset(initial_state=initial_state)
        self.history.clear()
        self._step_counter = 0
        self._record_initial_state()

    def get_history_df(self) -> pd.DataFrame:
        """Convert history log into a formatted Pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                "Step", "Timestamp", "From State", "Event", "To State", "Output Emitted", "Status", "Details"
            ])

        df = pd.DataFrame(self.history)
        df = df.rename(columns={
            "step": "Step",
            "timestamp": "Timestamp",
            "from_state": "From State",
            "event": "Event",
            "to_state": "To State",
            "output": "Output Emitted",
            "status": "Status",
            "message": "Details",
        })
        return df

    def analyze_reachability(self) -> Dict[str, Any]:
        """
        Perform graph reachability analysis from initial state.
        Useful for detecting unreachable/orphaned states in custom user edits.
        """
        visited: Set[str] = set()
        queue: List[str] = [self.fst.initial_state]

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                for (from_st, _), to_st in self.fst.transition_table.items():
                    if from_st == node and to_st not in visited:
                        queue.append(to_st)

        unreachable = self.fst.states - visited
        dead_ends = {
            s for s in self.fst.states
            if s not in self.fst.terminal_states and len(self.fst.get_valid_events(s)) == 0
        }

        return {
            "reachable_states": visited,
            "unreachable_states": unreachable,
            "dead_end_states": dead_ends,
            "is_fully_connected": len(unreachable) == 0,
        }
