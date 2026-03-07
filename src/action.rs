//! Action trait definition for the MCTS framework.
//!
//! This module defines the core interface that all actions must implement.
//! The Action trait is essential for the MCTS algorithm because it provides
//! the mechanism to transition between states - without actions, there would
//! be no tree to search.
//!
//! ## Why a trait?
//!
//! Using a trait allows the MCTS implementation to be domain-agnostic.
//! The same MCTS algorithm can work with different action types
//! (grid movements, card game moves, robot actions) as long as they
//! implement this interface.

use crate::state::State;

/// The Action trait represents a decision or move that can be taken in a state.
///
/// Any type implementing this trait can be used with the MCTS algorithm.
/// The trait is object-safe, meaning we can create trait objects
/// (`Box<dyn Action>`) to store heterogeneous action types.
pub trait Action {
    /// Applies this action to a given state, producing a new state.
    ///
    /// This represents taking the action in the current state, which results
    /// in a transition to a new state. The original state should not be modified.
    ///
    /// # Arguments
    /// * `state` - The current state to apply the action to
    ///
    /// # Returns
    /// A new state representing the result of taking this action
    fn apply_to(&self, state: &dyn State) -> Box<dyn State>;

    /// Returns the name of this action for display and debugging purposes.
    ///
    /// This should return a static string that identifies the action type.
    fn get_name(&self) -> &'static str;

    /// Creates a heap-allocated clone of this action as a trait object.
    ///
    /// This enables cloning of `Box<dyn Action>` through the `clone_box` method.
    /// We need this because we can't directly derive Clone on trait objects.
    fn clone_box(&self) -> Box<dyn Action>;
}

// Implement Clone for Box<dyn Action> by delegating to clone_box.
// This allows us to clone trait objects when needed (e.g., storing in tree).
impl Clone for Box<dyn Action> {
    fn clone(&self) -> Self {
        return self.clone_box();
    }
}
