//! Reward type definition for the MCTS framework.
//!
//! This module defines the Reward type alias which represents the value
//! returned by evaluating a state. Using a type alias allows for easy
//! modification of the underlying numeric type if needed (e.g., changing
//! from f64 to a more precise decimal type).
//!
//! ## Reward Design
//!
//! In reinforcement learning and game tree search, rewards typically:
//! - Are positive for good outcomes (winning, reaching goals)
//! - Are negative for bad outcomes (losing, hitting obstacles)
//! - Are zero for intermediate states
//!
//! The specific values depend on the domain. In this implementation,
//! we use f64 for flexibility and precision.

/// Type alias for reward values returned by state evaluation.
///
/// Using f64 provides:
/// - Good precision for most game scoring scenarios
/// - Compatibility with standard math operations
/// - Easy integration with neural networks if needed later
pub type Reward = f64;
