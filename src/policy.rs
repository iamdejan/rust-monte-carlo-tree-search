use rand::RngExt;

use crate::{action::Action, state::State};

pub type RolloutPolicy = fn(state: &dyn State) -> Option<Box<dyn Action>>;

/// Selects a random action from the legal actions available in the given state.
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
    let mut actions = state.get_legal_actions();
    if actions.is_empty() {
        return None;
    }

    // Use gen_range to get a random index, then remove that element to transfer ownership
    let mut rng = rand::rng();
    let idx = rng.random_range(0..actions.len());
    return Some(actions.remove(idx));
}
