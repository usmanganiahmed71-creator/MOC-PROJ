# 🎮 Game Environment State Machine & Path Inspector

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, rigorous web application and inspection tool that applies formal **Automata Theory** — a **Deterministic Finite Automaton (DFA)** for state legality checking and a **Mealy Finite State Transducer (FST)** for event-driven output/action dispatch — to manage, validate, and visualize Non-Player Character (NPC) behavior states in video games.

Built as a college group project for semester evaluation and codebase sharing.

---

## 🌟 Key Features

1. **Formal Automata Theory Engine (`src/automata.py`)**:
   - Strictly implements the formal 5-tuple DFA $M = (Q, \Sigma, \delta, q_0, F)$ and 6-tuple Mealy FST $M = (Q, \Sigma, \Gamma, \delta, \omega, q_0)$.
   - Outputs are bound directly to transitions: $\delta(q, \sigma) \to (q', \gamma)$.
   - Custom `IllegalTransitionError` raised on undefined transitions or attempts to leave terminal states.
2. **Runtime Simulation Wrapper (`src/state_machine.py`)**:
   - Safe execution wrapper preventing crashes on illegal inputs.
   - Enforces the core rule: on an illegal transition, the NPC **stays in its current state** without state deviation.
   - Maintains an immutable chronological audit history log with millisecond timestamps and Pandas DataFrame integration.
3. **Dynamic Graph Visualizer (`src/visualizer.py`)**:
   - Real-time Graphviz directed graph generation.
   - Highlights the active state in glowing green (`#10b981`), terminal/absorbing states in red (`#ef4444`), and marks the start state ($q_0 = \text{Idle}$).
   - Edges clearly labeled with `event / output` formatting.
4. **Multi-Tab Streamlit Dashboard (`app.py`)**:
   - **Tab 1: Live Interactive Simulator**: Real-time event buttons, transition mode toggles, instant diagnostic feedback, and searchable execution history with CSV export.
   - **Tab 2: Batch Sequence Validator**: Multi-event string verification ($w \in L(M)$ vs $w \notin L(M)$) with step-by-step trace tables.
   - **Tab 3: Formal Theory & Documentation**: Academic LaTeX formulations, Mealy vs. Moore comparative analysis, and complete transition matrix.
   - **Sidebar: Custom State & Transition Editor**: Add states/events/transitions with automatic reachability and orphaned-state detection.

---

## 📐 Reference Automaton Specification

### Formal Definition
- **States ($Q$):** `Idle`, `Patrol`, `Chase`, `Attack`, `Dead`
- **Initial State ($q_0$):** `Idle`
- **Terminal / Absorbing State:** `Dead` (No outgoing transitions defined; any event fired while in `Dead` is rejected)
- **Input Alphabet ($\Sigma$):**
  `StartPatrol`, `PlayerSpotted`, `PlayerLost`, `PlayerAdjacent`, `PlayerOutOfRange`, `TakeDamage`, `HealthZero`
- **Output Alphabet ($\Gamma$):**
  `begin_patrol_route`, `play_alert_sound`, `resume_patrol`, `play_attack_animation`, `resume_chase`, `play_death_animation`

### Transition Table: $\delta(q, \sigma) \to (q', \gamma)$

| Current State ($q$) | Event ($\sigma$) | Next State ($q'$) | Output Action ($\gamma$) | Behavior Type |
|---|---|---|---|---|
| `Idle` | `StartPatrol` | `Patrol` | `begin_patrol_route` | Operational |
| `Patrol` | `PlayerSpotted` | `Chase` | `play_alert_sound` | Operational |
| `Patrol` | `TakeDamage` | `Chase` | `play_alert_sound` | Operational |
| `Chase` | `PlayerLost` | `Patrol` | `resume_patrol` | Operational |
| `Chase` | `PlayerAdjacent` | `Attack` | `play_attack_animation` | Operational |
| `Attack` | `PlayerOutOfRange` | `Chase` | `resume_chase` | Operational |
| `Attack` | `PlayerLost` | `Patrol` | `resume_patrol` | Operational |
| `Idle` | `HealthZero` | `Dead` | `play_death_animation` | Terminal (Absorbing) |
| `Patrol` | `HealthZero` | `Dead` | `play_death_animation` | Terminal (Absorbing) |
| `Chase` | `HealthZero` | `Dead` | `play_death_animation` | Terminal (Absorbing) |
| `Attack` | `HealthZero` | `Dead` | `play_death_animation` | Terminal (Absorbing) |
| `Dead` | *Any Event* | **REJECTED** | `NONE` | Illegal Transition |

> **Note on Undefined Transitions:** Any pair $(q, \sigma)$ not listed above (such as `(Idle, PlayerAdjacent)`) is strictly undefined and rejected by the DFA legality engine.

---

## 📁 Directory Structure

```
game-state-inspector/
├── .streamlit/
│   └── config.toml           # Theme configuration (cybernetic dark mode)
├── assets/
│   └── custom.css            # Glassmorphic UI styling, badges, and animations
├── src/
│   ├── __init__.py           # Package exports
│   ├── automata.py           # Core DFA & Mealy FST formal classes
│   ├── visualizer.py         # Graphviz visualizer with active/terminal highlighting
│   └── state_machine.py      # Runtime simulation wrapper and history audit logger
├── tests/
│   ├── __init__.py
│   └── test_automata.py      # Full Pytest test suite
├── .gitignore                # Git ignore configuration
├── README.md                 # Project documentation and guide
├── requirements.txt          # Python dependencies
└── app.py                    # Streamlit web application dashboard
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed.

### 2. Installation
Clone or navigate to the project directory:
```bash
git clone <your-repo-url>
cd proj
```

Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

> ⚠️ **Note on Graphviz:**
> - Inside the Streamlit dashboard, diagrams are rendered directly in the web browser using WebAssembly / Viz.js without requiring the OS Graphviz binary.
> - If you plan to use `graphviz` Python library to generate standalone PNG/PDF images from the command line outside of Streamlit, install the system binary:
>   - **Windows:** `winget install Graphviz.Graphviz` or `choco install graphviz`
>   - **macOS:** `brew install graphviz`
>   - **Ubuntu/Debian:** `sudo apt install graphviz`

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite with verbose output:
```bash
python -m pytest tests/ -v
```

### Test Coverage Highlights:
- ✅ **Canonical Path Verification:** `Idle -> Patrol -> Chase -> Attack -> Dead`
- ✅ **Illegal Transition Rejection:** Direct `Idle + PlayerAdjacent` rejected; state remains `Idle`
- ✅ **Absorbing Terminal State:** All events from `Dead` are rejected
- ✅ **Mealy Output Correctness:** Verifies outputs emitted across all defined transitions
- ✅ **Undefined Input Alphabet Handling:** Non-alphabet symbols raise `IllegalTransitionError` gracefully
- ✅ **Batch Validation & Traces:** Step-by-step diagnostic verification of sequences
- ✅ **Dynamic State/Transition Addition:** Extensibility test for custom FST configurations
- ✅ **FSM State Reset:** Proper state and history restoration

---

## 🎓 Teammate & Presentation Notes

For teammates preparing for project presentation or code review:
1. **Why Mealy over Moore?**
   - In video game event loops, actions (like playing an audio alert or triggering a hit animation) are triggered by *events*, not just by *being in a state*. Mealy transducers bind outputs to the transition $\delta(q, \sigma) \to (q', \gamma)$, avoiding state explosion.
2. **Determinism:**
   - Every defined $(q, \sigma)$ maps to exactly one $(q', \gamma)$, guaranteeing predictable, deterministic NPC behavior.
3. **Safety & Robustness:**
   - Unhandled edge cases do not crash the game loop. The `NPCStateMachine` catches `IllegalTransitionError`, preserves the active state, and logs the incident.
