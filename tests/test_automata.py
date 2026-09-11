"""
Comprehensive Unit Test Suite for Formal Automata & NPC State Machine.

Tests cover:
- Valid full path: Idle -> Patrol -> Chase -> Attack -> Dead
- Illegal direct transition: Idle -> Attack (rejected, stays in Idle)
- Dead state is absorbing/terminal: any event from Dead is rejected
- FST output correctness across multiple distinct transitions
- Undefined events (not in Sigma) handled gracefully with IllegalTransitionError
- Batch sequence verification (valid vs invalid traces)
- Dynamic custom transitions & state machine reset functionality
"""

import pytest
from src.automata import DFA, FST, IllegalTransitionError
from src.state_machine import NPCStateMachine, build_default_fst
from src.visualizer import render_automaton_graph


class TestAutomataCore:
    """Test core formal DFA and Mealy FST implementations."""

    def test_default_initial_state(self):
        """Verify the state machine begins in the Idle state."""
        fsm = NPCStateMachine()
        assert fsm.current_state == "Idle"
        assert not fsm.is_dead
        assert fsm.get_valid_events() == ["HealthZero", "StartPatrol"]

    def test_valid_full_combat_path(self):
        """
        Test the complete valid canonical path:
        Idle -> Patrol -> Chase -> Attack -> Dead.
        """
        fsm = NPCStateMachine()

        # 1. Idle -> Patrol
        success, out, state = fsm.trigger("StartPatrol")
        assert success is True
        assert out == "begin_patrol_route"
        assert state == "Patrol"
        assert fsm.current_state == "Patrol"

        # 2. Patrol -> Chase
        success, out, state = fsm.trigger("PlayerSpotted")
        assert success is True
        assert out == "play_alert_sound"
        assert state == "Chase"
        assert fsm.current_state == "Chase"

        # 3. Chase -> Attack
        success, out, state = fsm.trigger("PlayerAdjacent")
        assert success is True
        assert out == "play_attack_animation"
        assert state == "Attack"
        assert fsm.current_state == "Attack"

        # 4. Attack -> Dead
        success, out, state = fsm.trigger("HealthZero")
        assert success is True
        assert out == "play_death_animation"
        assert state == "Dead"
        assert fsm.current_state == "Dead"
        assert fsm.is_dead is True

    def test_illegal_direct_transition_idle_to_attack(self):
        """
        Verify that an illegal transition (Idle + PlayerAdjacent) is rejected,
        does not crash, raises IllegalTransitionError at engine level,
        and keeps the NPC in its current state ('Idle').
        """
        fst = build_default_fst()
        with pytest.raises(IllegalTransitionError) as exc_info:
            fst.step("PlayerAdjacent")
        assert "Illegal transition" in str(exc_info.value) or "undefined" in str(exc_info.value).lower()
        assert fst.current_state == "Idle"

        # Test through wrapper
        fsm = NPCStateMachine()
        success, error_msg, state = fsm.trigger("PlayerAdjacent")
        assert success is False
        assert "undefined" in error_msg.lower() or "illegal" in error_msg.lower()
        assert state == "Idle"
        assert fsm.current_state == "Idle"

    def test_dead_is_absorbing_state(self):
        """
        Verify that Dead is strictly an absorbing/terminal state:
        NO outgoing transitions are allowed; any event fired while Dead is rejected.
        """
        fsm = NPCStateMachine()
        # Move to Dead from Idle directly
        fsm.trigger("HealthZero")
        assert fsm.current_state == "Dead"
        assert fsm.is_dead is True
        assert fsm.get_valid_events() == []

        # Attempt every possible event from Dead
        events_to_test = [
            "StartPatrol",
            "PlayerSpotted",
            "PlayerLost",
            "PlayerAdjacent",
            "PlayerOutOfRange",
            "TakeDamage",
            "HealthZero",
        ]

        for ev in events_to_test:
            success, msg, state = fsm.trigger(ev)
            assert success is False, f"Event '{ev}' should have failed from Dead."
            assert state == "Dead"
            assert fsm.current_state == "Dead"
            assert "terminal" in msg.lower() or "absorbing" in msg.lower()

    def test_fst_output_correctness_multiple_transitions(self):
        """
        Verify transducer output correctness for multiple distinct transitions:
        - (Idle, StartPatrol) -> begin_patrol_route
        - (Patrol, TakeDamage) -> play_alert_sound
        - (Patrol, PlayerSpotted) -> play_alert_sound
        - (Chase, PlayerLost) -> resume_patrol
        - (Chase, PlayerAdjacent) -> play_attack_animation
        - (Attack, PlayerOutOfRange) -> resume_chase
        - (Attack, PlayerLost) -> resume_patrol
        - (*, HealthZero) -> play_death_animation
        """
        fst = build_default_fst()

        # 1. Idle -> Patrol
        st, out = fst.step("StartPatrol")
        assert st == "Patrol"
        assert out == "begin_patrol_route"

        # 2. Patrol -> Chase via TakeDamage
        st, out = fst.step("TakeDamage")
        assert st == "Chase"
        assert out == "play_alert_sound"

        # 3. Chase -> Patrol via PlayerLost
        st, out = fst.step("PlayerLost")
        assert st == "Patrol"
        assert out == "resume_patrol"

        # 4. Patrol -> Chase via PlayerSpotted
        st, out = fst.step("PlayerSpotted")
        assert st == "Chase"
        assert out == "play_alert_sound"

        # 5. Chase -> Attack via PlayerAdjacent
        st, out = fst.step("PlayerAdjacent")
        assert st == "Attack"
        assert out == "play_attack_animation"

        # 6. Attack -> Chase via PlayerOutOfRange
        st, out = fst.step("PlayerOutOfRange")
        assert st == "Chase"
        assert out == "resume_chase"

        # 7. Chase -> Dead via HealthZero
        st, out = fst.step("HealthZero")
        assert st == "Dead"
        assert out == "play_death_animation"

    def test_undefined_event_not_in_alphabet_handled_gracefully(self):
        """
        Verify that passing an unknown event symbol (not in Sigma)
        raises an IllegalTransitionError and does not crash the state machine.
        """
        fst = build_default_fst()
        with pytest.raises(IllegalTransitionError) as exc_info:
            fst.step("JumpToHyperspace")
        assert "alphabet" in str(exc_info.value).lower()

        # Test through wrapper
        fsm = NPCStateMachine()
        success, err_msg, state = fsm.trigger("JumpToHyperspace")
        assert success is False
        assert state == "Idle"
        assert fsm.current_state == "Idle"
        assert "alphabet" in err_msg.lower()

    def test_combat_loop_repetition(self):
        """Test cycling between Chase and Attack multiple times."""
        fst = build_default_fst()
        fst.step("StartPatrol")      # -> Patrol
        fst.step("PlayerSpotted")    # -> Chase
        
        for _ in range(3):
            st, out = fst.step("PlayerAdjacent")
            assert st == "Attack"
            assert out == "play_attack_animation"

            st, out = fst.step("PlayerOutOfRange")
            assert st == "Chase"
            assert out == "resume_chase"

        assert fst.current_state == "Chase"

    def test_batch_sequence_validation(self):
        """Test batch sequence validation on DFA / FST."""
        fst = build_default_fst()

        # Valid sequence
        valid_seq = ["StartPatrol", "PlayerSpotted", "PlayerAdjacent", "HealthZero"]
        is_accepted, trace = fst.validate_with_outputs(valid_seq)
        assert is_accepted is True
        assert len(trace) == 4
        assert trace[0]["valid"] is True
        assert trace[0]["output"] == "begin_patrol_route"
        assert trace[-1]["to_state"] == "Dead"

        # Invalid sequence (Premature Attack)
        invalid_seq = ["StartPatrol", "PlayerAdjacent"]
        is_accepted, trace = fst.validate_with_outputs(invalid_seq)
        assert is_accepted is False
        assert len(trace) == 2
        assert trace[0]["valid"] is True
        assert trace[1]["valid"] is False
        assert "undefined" in trace[1]["error"].lower()

        # Invalid post-mortem event
        post_mortem_seq = ["HealthZero", "StartPatrol"]
        is_accepted, trace = fst.validate_with_outputs(post_mortem_seq)
        assert is_accepted is False
        assert len(trace) == 2
        assert trace[0]["valid"] is True
        assert trace[1]["valid"] is False
        assert "terminal" in trace[1]["error"].lower()

    def test_state_machine_reset(self):
        """Verify resetting the state machine clears history and restores starting state."""
        fsm = NPCStateMachine()
        fsm.trigger("StartPatrol")
        fsm.trigger("PlayerSpotted")
        assert fsm.current_state == "Chase"
        assert len(fsm.history) > 1

        fsm.reset()
        assert fsm.current_state == "Idle"
        assert len(fsm.history) == 1
        assert fsm.history[0]["event"] == "INITIALIZE"

    def test_custom_transition_addition(self):
        """Verify dynamically adding new states and transitions to the Mealy FST."""
        fst = build_default_fst()
        # Add custom state 'Stunned'
        fst.add_transition(
            from_state="Chase",
            event="FlashbangHit",
            to_state="Stunned",
            output="play_stunned_animation",
        )
        fst.add_transition(
            from_state="Stunned",
            event="StunRecover",
            to_state="Chase",
            output="regain_balance",
        )

        assert "Stunned" in fst.states
        assert "FlashbangHit" in fst.alphabet
        assert "play_stunned_animation" in fst.output_alphabet

        fst.step("StartPatrol")
        fst.step("PlayerSpotted")
        st, out = fst.step("FlashbangHit")
        assert st == "Stunned"
        assert out == "play_stunned_animation"

        st, out = fst.step("StunRecover")
        assert st == "Chase"
        assert out == "regain_balance"

    def test_visualizer_generation(self):
        """Verify Graphviz dot diagram generation succeeds without errors."""
        fst = build_default_fst()
        dot = render_automaton_graph(fst, current_state="Chase")
        source = dot.source
        assert "NPC_Automaton_FST" in source
        assert "Chase" in source
        assert "Dead" in source
        assert "PlayerSpotted / play_alert_sound" in source
