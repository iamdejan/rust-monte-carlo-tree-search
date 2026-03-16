# Rust Implementation of Monte-Carlo Tree Search

This is Rust implementation of Monte-Carlo Tree Search. The grid world is based on the [example from Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/monte-carlo-tree-search/), while the code structure and implementation is loosely based on [int8.io blog](https://int8.io/monte-carlo-tree-search-beginners-guide/) and [code](https://github.com/int8/gomcts), albeit with modifications due to the nature of the world and the characteristics of Rust language.

## Prerequisites

Before running this program, ensure you have the following installed:

1. **Pixi** - A package manager for Rust and other languages. Install it by following the official guide at [https://pixi.prefix.dev/latest/](https://pixi.prefix.dev/latest/).

2. **rust-docs-mcp** - An MCP server that provides comprehensive access to Rust crate documentation, source code analysis, dependency trees, and module structure visualization. Install it using one of the following methods:

   **Quick Install (recommended):**
   ```bash
   curl -sSL https://raw.githubusercontent.com/snowmead/rust-docs-mcp/main/install.sh | bash
   ```

   **Or via Cargo:**
   ```bash
   cargo install rust-docs-mcp
   ```

   For more installation options, see the [official installation guide](https://github.com/snowmead/rust-docs-mcp#installation).

3. **Download Dependencies** - Once Pixi is installed, run the following command to download all project dependencies:
   ```bash
   pixi install
   ```

4. **Setup Pre-commit** - Install the pre-commit hooks to automatically run linting and build checks before push:
   ```bash
   pixi run pre-commit install --hook-type pre-push
   ```

   > **Note:** Re-run this command whenever `.pre-commit-config.yaml` is changed to update the hooks.

5. **Build the project** - If you want to build the project without running it, run the following command:
   ```bash
   pixi run build
   ```

## Run the Program

Use the following command to build and run the project:
   ```bash
   pixi run start
   ```

## Grid World

This project demonstrates MCTS solving a grid-based navigation problem. The grid world is a classic reinforcement learning environment used to test decision-making algorithms.

### Grid Layout

The grid is a **3x4** layout (3 rows, 4 columns) with the following structure:

|   | Col 0 | Col 1 | Col 2 | Col 3 |
|---|-------|-------|-------|-------|
| **Row 0** |       |       |       | Goal (+1) |
| **Row 1** | Start | Blocked |   | Penalty (-1) |
| **Row 2** |       |       |       |       |

### Coordinate System

- **Rows** are indexed from top to bottom (0 to 2)
- **Columns** are indexed from left to right (0 to 3)
- Position (r, c) represents row `r` and column `c`

### Special Cells

| Cell Type | Position | Description |
|-----------|----------|-------------|
| **Start** | (1, 0) | The agent begins at row 1, column 0 (left side of middle row) |
| **Goal** | (0, 3) | Top-right corner - reaching this cell yields reward **+1** and ends the episode |
| **Penalty** | (1, 3) | Row 1, Column 3 - acts as a trap with reward **-1**, also ends the episode |
| **Blocked** | (1, 1) | Row 1, Column 1 - obstacle cell that cannot be entered |

### Terminal States

The episode ends (terminal state reached) when the agent arrives at:
- **Goal cell (0, 3)** - Success! Reward = +1
- **Penalty cell (1, 3)** - Failure! Reward = -1

At terminal states, no further actions are possible and the final reward is assigned.

### Actions

The agent can move in four cardinal directions:
- **Up** - Move one row up (decreases row index)
- **Down** - Move one row down (increases row index)
- **Left** - Move one column left (decreases column index)
- **Right** - Move one column right (increases column index)

Attempting to move outside the grid boundaries or into the blocked cell results in staying at the current position (invalid moves are rejected).

### Objective

The MCTS algorithm aims to find the optimal path from the starting position (1, 0) to the goal cell (0, 3) while avoiding the penalty cell (1, 3) and the blocked cell (1, 1).

## Available Commands

| Command | Description |
|---------|-------------|
| `pixi run start` | Build and run the MCTS program |
| `pixi run build` | Compile the Rust project |
| `pixi run test` | Run all unit tests |
| `pixi run fmt` | Format code using rustfmt |
| `pixi run lint` | Run clippy linter (depends on fmt) |
| `pixi run lint-fix` | Automatically fix linting issues |
| `pixi run clean` | Remove build artifacts |

## Project Tree Structure

```
Root/
├── .github/
│   └── workflows/
│       ├── pr-pipeline.yaml      # CI/CD pipeline for pull requests
│       └── trunk-pipeline.yaml   # CI/CD pipeline for trunk/main branch
├── .kilocode/                      # Kilo Code agent configuration and MCP settings
├── src/
│   ├── action.rs                 # Action trait definition for MCTS
│   ├── grid_world.rs             # Grid world domain implementation
│   ├── main.rs                   # Main entry point
│   ├── mcts.rs                   # Monte Carlo Tree Search algorithm
│   ├── policy.rs                 # Rollout policy for simulations
│   ├── position.rs               # Position type for grid coordinates
│   ├── reward.rs                 # Reward type definition
│   └── state.rs                  # State trait definition
├── .editorconfig                 # Editor configuration for consistent formatting
├── .gitattributes                # Git attributes (handles pixi.lock merging)
├── .gitignore                    # Git ignore patterns
├── AGENTS.md                     # Agent guidelines for code generation, debugging, and validation
├── Cargo.lock                    # Cargo dependency lock file
├── Cargo.toml                    # Cargo manifest (Rust project config)
├── LICENSE.txt                   # Public domain license (Unlicense)
├── pixi.lock                     # Pixi dependency lock file
├── pixi.toml                     # Pixi project configuration
└── README.md                     # This file
```

This section explains the purpose of each file in the repository:

### Root Directory Files

| File | Description |
|------|-------------|
| [`Cargo.toml`](Cargo.toml:1) | Cargo manifest defining the package name, version, edition, lints, and dependencies |
| [`Cargo.lock`](Cargo.lock:1) | Auto-generated lock file ensuring reproducible builds with exact dependency versions |
| [`pixi.toml`](pixi.toml:1) | Pixi project configuration defining workspace, tasks (build, test, lint), and dependencies |
| [`pixi.lock`](pixi.lock:1) | Pixi lock file for reproducible environments |
| [`LICENSE.txt`](LICENSE.txt:1) | The Unlicense - places the software in the public domain |
| [`README.md`](README.md:1) | Project documentation (this file) |
| [`.editorconfig`](.editorconfig:1) | Editor configuration ensuring consistent code style (4-space indentation for Rust) |
| [`.gitattributes`](.gitattributes:1) | Git attributes for handling pixi.lock as binary YAML |
| [`.gitignore`](.gitignore:1) | Git ignore patterns for build artifacts, IDE files, and pixi environments |
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1) | Pre-commit hooks configuration for running lint and format checks before commits |
| [`AGENTS.md`](AGENTS.md:1) | Agent guidelines containing rules for code generation, debugging, and validation |

### Source Files

| File | Description |
|------|-------------|
| [`src/main.rs`](src/main.rs:1) | Main entry point demonstrating MCTS solving a grid navigation problem |
| [`src/mcts.rs`](src/mcts.rs:1) | Core MCTS algorithm implementation with selection, expansion, simulation, and backpropagation phases |
| [`src/state.rs`](src/state.rs:1) | State trait defining the interface for game states (position, legal actions, evaluation, terminal check) |
| [`src/action.rs`](src/action.rs:1) | Action trait defining the interface for actions (apply to state, get name) |
| [`src/position.rs`](src/position.rs:1) | Position struct representing (row, column) coordinates in the grid |
| [`src/reward.rs`](src/reward.rs:1) | Reward type alias (f64) for state evaluation values |
| [`src/policy.rs`](src/policy.rs:1) | Default random rollout policy for MCTS simulations |
| [`src/grid_world.rs`](src/grid_world.rs:1) | Grid world domain implementation: 3x4 grid with goal (+1), penalty (-1), and blocked cell |

### Configuration Directories

| Directory | Description |
|-----------|-------------|
| [`.github/workflows/`](.github/workflows/) | CI/CD pipelines: `pr-pipeline.yaml` runs lint and tests on pull requests; `trunk-pipeline.yaml` runs on main branch |
| [`.kilocode/`](.kilocode/) | Kilo Code agent configuration and MCP settings (mcp.json) |
