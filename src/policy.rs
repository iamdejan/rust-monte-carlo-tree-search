//! Rollout policy implementation for MCTS simulations.
//!
//! This module provides the default random rollout policy used during
//! the simulation phase of MCTS. The rollout policy decides which
//! action to take when the tree search "plays out" a random game.
//!
//! ## Why Random Rollout?
//!
//! The simulation/rollout phase in MCTS uses random playouts because:
//! 1. It's computationally efficient (no tree search needed)
//! 2. With enough samples, the average outcome converges to the true value
//! 3. It provides an unbiased estimate of a state's worth
//!
//! More sophisticated policies can be used (like heuristics or learned policies)
//! to improve convergence speed, but random is the standard baseline.

use rand::RngExt;

use crate::{action::Action, state::State};

/// Type alias for a rollout policy function.
///
/// A rollout policy takes a state and returns an action to take.
/// It returns None if no legal actions are available (shouldn't happen
/// in normal play from non-terminal states).
pub type RolloutPolicy = fn(state: &dyn State) -> Option<Box<dyn Action>>;

/// Selects a random action from the legal actions available in the given state.
///
/// This is the default rollout policy used during MCTS simulations.
/// It provides unbiased random sampling which, over many simulations,
/// gives accurate estimates of state values.
///
/// This function performs the following steps:
/// 1. Retrieves all legal actions from the state via `get_legal_actions()`
/// 2. Checks if any legal actions exist; returns `None` if the list is empty
/// 3. Generates a random index into the actions slice
/// 4. Removes and returns the action at that index, transferring ownership to the caller
///
/// # Arguments
/// * `state` - A reference to a State object that provides legal actions
///
/// # Returns
/// * `Option<Box<dyn Action>>` - Some action owned by the caller, or None if no legal actions exist
///
/// # Panics
/// Panics if the random number generator fails (extremely rare)
pub fn default(state: &dyn State) -> Option<Box<dyn Action>> {
    // Get all legal actions from the current state
    let mut actions = state.get_legal_actions();

    // No legal actions available - this shouldn't happen in normal play
    // but we handle it gracefully
    if actions.is_empty() {
        return None;
    }

    // Use the random number generator to select an index
    // We use gen_range to pick uniformly from [0, actions.len())
    let mut rng = rand::rng();
    let idx = rng.random_range(0..actions.len());

    // Remove and return the selected action
    // We use remove() to transfer ownership to the caller
    return Some(actions.remove(idx));
}
