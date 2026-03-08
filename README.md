# Rust Implementation of Monte-Carlo Tree Search

This is Rust implementation of Monte-Carlo Tree Search. The grid world is based on the [example from Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/monte-carlo-tree-search/), while the code structure and implementation is loosely based on [int8.io blog](https://int8.io/monte-carlo-tree-search-beginners-guide/) and [code](https://github.com/int8/gomcts), albeit with modifications due to the nature of the world and the characteristics of Rust language.

## Prerequisites

To run this program, you need the following:

1. **Install Pixi** - A fast Python package manager built in Rust
   ```bash
   # On macOS/Linux using curl
   curl -fsSL https://pixi.sh/install.sh | bash
   
   # Or using pip
   pip install pixi
   ```

2. **Run the program**
   ```bash
   pixi run start
   ```

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
rust-monte-carlo-tree-search/
├── .github/
│   └── workflows/
│       ├── pr-pipeline.yaml      # CI/CD pipeline for pull requests
│       └── trunk-pipeline.yaml   # CI/CD pipeline for trunk/main branch
├── .kilocode/
│   └── rules/
│       ├── code_generation.md    # Guidelines for AI code generation
│       └── code_validation.md    # Guidelines for code validation
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
├── Cargo.lock                    # Cargo dependency lock file
├── Cargo.toml                    # Cargo manifest (Rust project config)
├── LICENSE.txt                   # Public domain license (Unlicense)
├── pixi.lock                     # Pixi dependency lock file
├── pixi.toml                     # Pixi project configuration
└── README.md                     # This file
```

## File Descriptions

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
| [`.kilocode/rules/`](.kilocode/rules/) | Kilo Code agent guidelines: code generation rules and code validation procedures |
