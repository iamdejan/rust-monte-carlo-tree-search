//! State trait definition for the MCTS framework.
//!
//! This module defines the core interface that all game states must implement.
//! The State trait provides the MCTS algorithm with everything it needs to explore
//! and evaluate the game tree:
//! - What actions are available (`get_legal_actions`)
//! - What the current state is (`get_current_position`)
//! - How good is this state (evaluate)
//! - Is the game over (`is_game_ended`)
//!
//! ## Why a trait?
//!
//! Using a trait makes the MCTS implementation reusable across different domains.
//! The same MCTS search can work with chess, go, poker, or grid worlds
//! as long as they provide a State implementation.

use crate::{action::Action, position::Position, reward::Reward};

/// The State trait represents a complete snapshot of the game environment.
///
/// Any type implementing this trait can be used with the MCTS algorithm.
/// It provides all the necessary interfaces for the search to explore
/// possible futures and evaluate their desirability.
///
/// The trait is object-safe, allowing us to create `Box<dyn State>` to
/// handle states polymorphically.
pub trait State {
    /// Returns the current position/position of the agent in this state.
    ///
    /// This is used by the MCTS to understand the current configuration
    /// and to determine what actions are available.
    ///
    /// # Returns
    /// A Position representing the agent's location
    fn get_current_position(&self) -> Position;

    /// Updates the agent's position to a new location.
    ///
    /// This is used when applying an action to create a new state.
    /// The implementation should validate the new position and reject
    /// invalid moves (e.g., moving outside boundaries or into obstacles).
    ///
    /// # Arguments
    /// * `new_position` - The desired new position
    fn update_current_position(&mut self, new_position: Position);

    /// Evaluates this state and returns a reward value.
    ///
    /// The reward quantifies how good this state is for the agent.
    /// Higher rewards are better. Common patterns:
    /// - Positive reward for winning/achieving goal
    /// - Negative reward for losing/hitting obstacle
    /// - Zero reward for intermediate states
    ///
    /// # Returns
    /// A Reward value (typically f64) representing the state's value
    fn evaluate(&self) -> Reward;

    /// Returns all legal actions that can be taken from this state.
    ///
    /// Legal actions are those that result in a valid new state.
    /// This filters out actions that would break the rules (e.g.,
    /// moving outside the board, playing an invalid card).
    ///
    /// # Returns
    /// A vector of boxed Actions representing all valid moves
    fn get_legal_actions(&self) -> Vec<Box<dyn Action>>;

    /// Checks whether the game has ended in this state.
    ///
    /// Terminal states are important for MCTS because:
    /// 1. No further actions can be taken
    /// 2. The reward is final (no need for further simulation)
    /// 3. We don't expand from terminal states
    ///
    /// # Returns
    /// true if the game is over, false otherwise
    fn is_game_ended(&self) -> bool;

    /// Creates a heap-allocated clone of this state as a trait object.
    ///
    /// This enables cloning of `Box<dyn State>` through the `clone_box` method.
    /// We need this because we can't directly derive Clone on trait objects,
    /// but MCTS needs to clone states for simulation.
    fn clone_box(&self) -> Box<dyn State>;
}

// Implement Clone for Box<dyn State> by delegating to clone_box.
// This allows us to clone trait objects when needed (e.g., during rollout).
impl Clone for Box<dyn State> {
    fn clone(&self) -> Self {
        return self.clone_box();
    }
}
