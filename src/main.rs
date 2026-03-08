//! Main entry point for the Monte Carlo Tree Search Grid World demo.
//!
//! This program demonstrates the MCTS algorithm solving a simple grid navigation problem.
//! The agent must find the optimal path from the starting position (1, 0) to the goal
//! at (0, 3) while avoiding the penalty cell at (1, 3) and blocked cell at (1, 1).
//!
//! ## How it Works
//!
//! 1. Create an initial `GridWorldState` at position (1, 0)
//! 2. Run MCTS search with 1000 simulations
//! 3. Get the best action from the search
//! 4. Apply the action to get the new state
//! 5. Repeat until the game ends (goal/penalty reached) or max steps exceeded
//!
//! ## Expected Output
//!
//! The algorithm should find a path to the goal. A typical good path might be:
//! Up -> Right -> Right -> Right (to reach the goal at position 0,3)

/// Module declarations for all components of the MCTS system.
/// Each module implements a specific aspect of the algorithm.
mod action;
mod grid_world;
mod mcts;
mod policy;
mod position;
mod reward;
mod state;

use crate::{grid_world::GridWorldState, state::State};

/// Maximum number of steps the agent can take before giving up.
///
/// This prevents infinite loops in case the algorithm fails to find
/// a terminal state. In practice, the grid is small enough that the
/// goal should be reached well before this limit.
const MAX_STEPS: i8 = 50;

fn main() {
    // Create the initial state at position (1, 0)
    let mut state: Box<dyn State> = Box::new(GridWorldState::new());

    // Track number of steps taken
    let mut steps = 0;

    // Main game loop: keep playing until game ends or we hit step limit
    while steps < MAX_STEPS {
        // Check if we've reached a terminal state (goal or penalty)
        if state.is_game_ended() {
            println!(
                "Game is ended! Step: {}, state={:#?}",
                steps,
                state.get_current_position()
            );
            break;
        }

        // Run MCTS to find the best action from current state
        // We use 1000 simulations for good accuracy
        let chosen_action = mcts::search(state.clone(), policy::default, 1000);

        // Print the chosen action for visualization
        println!("{}", chosen_action.get_name());

        // Apply the action to get the new state
        let new_state: Box<dyn State> = chosen_action.apply_to(state.as_ref());

        // Print the resulting position for debugging
        println!(
            "New state's position: {:#?}",
            new_state.get_current_position()
        );

        // Move to the new state and continue
        state = new_state;

        steps += 1;
    }

    // Check if we hit the step limit without reaching terminal state
    if steps >= MAX_STEPS {
        println!("Reached max steps without reaching terminal");
    }
}
