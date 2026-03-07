//! Position type for representing coordinates in the grid world.
//!
//! This module provides a simple 2D position type that uses row-column
//! coordinates (r, c). This is a common representation for grid-based
//! problems where we need to track the agent's location.
//!
//! ## Coordinate System
//!
//! The coordinate system uses:
//! - `r` (row): Increases downward (0 is the top row)
//! - `c` (column): Increases rightward (0 is the leftmost column)
//!
//! This matches how matrices are typically indexed in computer science
//! and how 2D arrays are laid out in memory.

/// Represents a position in 2D space using row and column coordinates.
///
/// This struct is used to represent positions in the grid world.
/// It derives common traits like Clone, Copy, `PartialEq`, Eq, and Debug
/// for convenience in testing and comparison.
///
/// # Example
/// ```
/// let pos = Position { r: 0, c: 0 };  // Top-left corner
/// let pos2 = Position { r: 2, c: 3 }; // Bottom-right in a 3x4 grid
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Position {
    /// The row coordinate (increases downward)
    pub r: i8,
    /// The column coordinate (increases rightward)
    pub c: i8,
}

impl Position {
    /// Adds two positions together, returning the result.
    ///
    /// This is used to compute new positions by adding a delta (direction)
    /// to a current position. For example, moving "right" adds (0, 1) to
    /// the current position.
    ///
    /// # Arguments
    /// * `other` - The position delta to add (e.g., {r: 1, c: 0} for "down")
    ///
    /// # Returns
    /// A new Position representing the sum of the two positions
    ///
    /// # Example
    /// ```
    /// let current = Position { r: 0, c: 0 };
    /// let delta = Position { r: 1, c: 0 };  // Move down
    /// let new_pos = current.add(delta);      // Result: {r: 1, c: 0}
    /// ```
    pub const fn add(&self, other: Self) -> Self {
        let new_r = self.r + other.r;
        let new_c = self.c + other.c;
        return Self { r: new_r, c: new_c };
    }
}
