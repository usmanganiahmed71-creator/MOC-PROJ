"""
Formal Automata Theory Core Engine.

This module implements:
1. Deterministic Finite Automaton (DFA) defined as a 5-tuple:
   M_DFA = (Q, Sigma, delta, q_0, F)
   where:
   - Q: Finite set of states
   - Sigma: Finite input alphabet of events
   - delta: Transition function delta: Q x Sigma -> Q
   - q_0: Initial start state (q_0 in Q)
   - F: Set of accept/final states (F subseteq Q)

2. Mealy Finite State Transducer (FST) defined as a 6-tuple:
   M_FST = (Q, Sigma, Gamma, delta, omega, q_0)
   where:
   - Q, Sigma, delta, q_0 are as defined above
   - Gamma: Finite output alphabet
   - omega: Output function omega: Q x Sigma -> Gamma
     (or unified transition mapping delta: Q x Sigma -> Q x Gamma)

Outputs in a Mealy machine are directly bound to transitions (delta(q, sigma) -> (q', gamma)),
making it an ideal formalism for event-driven video game NPC behavior dispatch.
"""

from typing import Dict, List, Optional, Set, Tuple, Any


class IllegalTransitionError(Exception):
    """
    Raised when an undefined or prohibited transition is triggered in the automaton.

    Attributes:
        current_state: The state from which the transition was attempted.
        event: The event that was received.
        message: Human-readable explanation of why the transition failed.
    """

    def __init__(self, current_state: str, event: str, message: Optional[str] = None):
        self.current_state = current_state
        self.event = event
        self.message = message or f"Illegal transition: Event '{event}' is undefined from state '{current_state}'."
        super().__init__(self.message)


class DFA:
    """
    Deterministic Finite Automaton (DFA).

    Formal Model:
        M = (Q, Sigma, delta, q_0, F)
    """

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transition_table: Dict[Tuple[str, str], str],
        initial_state: str,
        accept_states: Optional[Set[str]] = None,
        terminal_states: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the Deterministic Finite Automaton.

        Args:
            states (Set[str]): Finite set of states (Q).
            alphabet (Set[str]): Finite set of input symbols/events (Sigma).
            transition_table (Dict[Tuple[str, str], str]): delta: Q x Sigma -> Q.
            initial_state (str): Start state q_0 in Q.
            accept_states (Optional[Set[str]]): Accept states F subseteq Q. Defaults to all valid non-error states.
            terminal_states (Optional[Set[str]]): Absorbing/terminal states with no outgoing transitions (e.g. {'Dead'}).

        Raises:
            ValueError: If initial_state or accept_states are not subsets of Q.
        """
        if initial_state not in states:
            raise ValueError(f"Initial state '{initial_state}' must be a member of states Q: {states}")

        self.states: Set[str] = set(states)
        self.alphabet: Set[str] = set(alphabet)
        self.transition_table: Dict[Tuple[str, str], str] = dict(transition_table)
        self.initial_state: str = initial_state
        self.accept_states: Set[str] = set(accept_states) if accept_states is not None else set(states)
        self.terminal_states: Set[str] = set(terminal_states) if terminal_states is not None else set()
        self.current_state: str = initial_state

        # Validate transition table consistency
        for (q, sigma), q_next in self.transition_table.items():
            if q not in self.states:
                raise ValueError(f"Transition source state '{q}' not in Q.")
            if sigma not in self.alphabet:
                raise ValueError(f"Transition event '{sigma}' not in Sigma.")
            if q_next not in self.states:
                raise ValueError(f"Transition target state '{q_next}' not in Q.")
            if q in self.terminal_states:
                raise ValueError(f"Terminal state '{q}' cannot have outgoing transitions.")

    def step(self, event: str) -> str:
        """
        Execute a single transition delta(current_state, event).

        Args:
            event (str): The input event sigma in Sigma.

        Returns:
            str: The next state q' in Q.

        Raises:
            IllegalTransitionError: If the event is not in alphabet or transition is undefined.
        """
        if self.current_state in self.terminal_states:
            raise IllegalTransitionError(
                self.current_state,
                event,
                f"State '{self.current_state}' is terminal/absorbing. No outgoing transitions are permitted."
            )

        if event not in self.alphabet:
            raise IllegalTransitionError(
                self.current_state,
                event,
                f"Event '{event}' is not in the automaton alphabet Sigma: {sorted(list(self.alphabet))}."
            )

        key = (self.current_state, event)
        if key not in self.transition_table:
            raise IllegalTransitionError(
                self.current_state,
                event,
                f"Undefined transition delta('{self.current_state}', '{event}')."
            )

        next_state = self.transition_table[key]
        self.current_state = next_state
        return next_state

    def is_valid_transition(self, state: str, event: str) -> bool:
        """Check if a transition is valid from the given state without altering machine state."""
        if state in self.terminal_states:
            return False
        if event not in self.alphabet:
            return False
        return (state, event) in self.transition_table

    def get_valid_events(self, state: Optional[str] = None) -> List[str]:
        """
        Return the list of valid outgoing events for a given state (defaults to current_state).
        """
        target_state = state if state is not None else self.current_state
        if target_state in self.terminal_states:
            return []
        return sorted([sigma for (q, sigma) in self.transition_table.keys() if q == target_state])

    def reset(self, initial_state: Optional[str] = None) -> None:
        """Reset the automaton to its initial state or a specified valid state."""
        if initial_state is not None:
            if initial_state not in self.states:
                raise ValueError(f"Cannot reset to unknown state '{initial_state}'.")
            self.current_state = initial_state
        else:
            self.current_state = self.initial_state

    def validate(
        self, sequence: List[str], start_state: Optional[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a sequence of events from a start state through the DFA.

        Args:
            sequence (List[str]): Sequence of event symbols [w_1, w_2, ..., w_n].
            start_state (Optional[str]): Starting state for validation (defaults to initial_state).

        Returns:
            Tuple[bool, List[Dict[str, Any]]]:
                - is_accepted: True if the entire sequence successfully traverses defined transitions
                  and ends in an accept state F.
                - trace: Step-by-step diagnostic trace of each transition.
        """
        curr = start_state if start_state is not None else self.initial_state
        if curr not in self.states:
            raise ValueError(f"Start state '{curr}' not in states Q.")

        trace: List[Dict[str, Any]] = []
        is_valid = True

        for idx, event in enumerate(sequence):
            clean_event = event.strip()
            step_record: Dict[str, Any] = {
                "step": idx + 1,
                "event": clean_event,
                "from_state": curr,
                "to_state": None,
                "valid": False,
                "error": None,
                "is_terminal": curr in self.terminal_states,
            }

            if curr in self.terminal_states:
                step_record["error"] = f"State '{curr}' is terminal/absorbing. Cannot process '{clean_event}'."
                trace.append(step_record)
                is_valid = False
                break

            if clean_event not in self.alphabet:
                step_record["error"] = f"Event '{clean_event}' not in alphabet Sigma."
                trace.append(step_record)
                is_valid = False
                break

            key = (curr, clean_event)
            if key not in self.transition_table:
                step_record["error"] = f"Undefined transition delta('{curr}', '{clean_event}')."
                trace.append(step_record)
                is_valid = False
                break

            next_st = self.transition_table[key]
            step_record["to_state"] = next_st
            step_record["valid"] = True
            step_record["is_terminal"] = next_st in self.terminal_states
            trace.append(step_record)
            curr = next_st

        is_accepted = is_valid and (curr in self.accept_states)
        return is_accepted, trace


class FST(DFA):
    """
    Mealy Finite State Transducer (FST).

    Formal Model:
        M_FST = (Q, Sigma, Gamma, delta, omega, q_0)
    where:
        Gamma: Finite output alphabet
        omega: Output function omega: Q x Sigma -> Gamma
               (unified as delta: Q x Sigma -> Q x Gamma)

    Transitions are defined such that firing an event triggers state change AND emits an output.
    """

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        output_alphabet: Set[str],
        transitions: Dict[Tuple[str, str], Tuple[str, str]],
        initial_state: str,
        accept_states: Optional[Set[str]] = None,
        terminal_states: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize Mealy Finite State Transducer.

        Args:
            states (Set[str]): States Q.
            alphabet (Set[str]): Input event alphabet Sigma.
            output_alphabet (Set[str]): Action/trigger output alphabet Gamma.
            transitions (Dict[Tuple[str, str], Tuple[str, str]]):
                Mapping (q, sigma) -> (q_next, gamma).
            initial_state (str): Start state q_0 in Q.
            accept_states (Optional[Set[str]]): Accept states F.
            terminal_states (Optional[Set[str]]): Terminal states (e.g. {'Dead'}).
        """
        # Split transitions into delta and omega tables
        delta_table: Dict[Tuple[str, str], str] = {}
        self.output_table: Dict[Tuple[str, str], str] = {}

        for (q, sigma), (q_next, gamma) in transitions.items():
            delta_table[(q, sigma)] = q_next
            self.output_table[(q, sigma)] = gamma

        super().__init__(
            states=states,
            alphabet=alphabet,
            transition_table=delta_table,
            initial_state=initial_state,
            accept_states=accept_states,
            terminal_states=terminal_states,
        )

        self.output_alphabet: Set[str] = set(output_alphabet)

        # Validate output alphabet
        for (q, sigma), gamma in self.output_table.items():
            if gamma not in self.output_alphabet:
                raise ValueError(f"Output '{gamma}' for transition ({q}, {sigma}) not in Gamma: {self.output_alphabet}")

    def step(self, event: str) -> Tuple[str, str]:
        """
        Execute a transition in the Mealy FST: (current_state, event) -> (next_state, output).

        Args:
            event (str): Input event sigma.

        Returns:
            Tuple[str, str]: (next_state, output_action).

        Raises:
            IllegalTransitionError: If the transition is undefined or state is terminal.
        """
        from_state = self.current_state
        # Perform base DFA transition check and state update
        next_state = super().step(event)
        output = self.output_table[(from_state, event)]
        return next_state, output

    def get_transition_output(self, state: str, event: str) -> Optional[str]:
        """Return the output associated with (state, event) or None if undefined."""
        return self.output_table.get((state, event))

    def validate_with_outputs(
        self, sequence: List[str], start_state: Optional[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a sequence of events and record the associated transducer outputs for each step.

        Args:
            sequence (List[str]): List of event strings.
            start_state (Optional[str]): Optional starting state.

        Returns:
            Tuple[bool, List[Dict[str, Any]]]: Acceptance boolean and enriched trace records.
        """
        is_accepted, trace = super().validate(sequence, start_state=start_state)
        for record in trace:
            if record["valid"]:
                from_st = record["from_state"]
                ev = record["event"]
                record["output"] = self.output_table.get((from_st, ev), "none")
            else:
                record["output"] = "REJECTED (no output emitted)"
        return is_accepted, trace

    def add_transition(self, from_state: str, event: str, to_state: str, output: str) -> None:
        """
        Add or update a transition (from_state, event) -> (to_state, output).
        Dynamically updates Q, Sigma, and Gamma if new elements are introduced.
        """
        if from_state in self.terminal_states:
            raise ValueError(f"Cannot add outgoing transition from terminal state '{from_state}'.")

        self.states.add(from_state)
        self.states.add(to_state)
        self.alphabet.add(event)
        self.output_alphabet.add(output)
        self.accept_states.add(to_state)

        self.transition_table[(from_state, event)] = to_state
        self.output_table[(from_state, event)] = output

    def remove_transition(self, from_state: str, event: str) -> bool:
        """Remove a transition if it exists. Returns True if removed, False otherwise."""
        key = (from_state, event)
        if key in self.transition_table:
            del self.transition_table[key]
            del self.output_table[key]
            return True
        return False
