mod position;
mod action;
mod state;
mod reward;

use crate::position::Position;
use crate::action::Action;
use crate::state::State;
use crate::reward::Reward;

#[derive(Debug)]
enum GridWorldAction {
    Up,
    Down,
    Left,
    Right,
}

impl GridWorldAction {
    const fn delta(&self) -> Position {
        match self {
            Self::Up => return Position { r: -1, c: 0 },
            Self::Down => return Position { r: 1, c: 0 },
            Self::Left => return Position { r: 0, c: -1 },
            Self::Right => return Position { r: 0, c: 1 },
        }
    }

    /// Returns the name of the action as a string slice.
    ///
    /// # Returns
    /// * `&'static str` - The name of the action variant (e.g., "Up", "Down", "Left", "Right")
    const fn name(&self) -> &'static str {
        match self {
            Self::Up => return "Up",
            Self::Down => return "Down",
            Self::Left => return "Left",
            Self::Right => return "Right",
        }
    }
}

impl Action for GridWorldAction {
    fn apply_to(&self, state: &mut dyn State) {
        let current_position = state.get_current_position();
        let delta = self.delta();
        let new_position = current_position.add(delta);
        state.update_current_position(new_position);
    }

    fn get_name(&self) -> &'static str {
        return self.name();
    }
}

#[derive(Debug)]
struct GridWorldState {
    pub current_position: Position,
}

impl GridWorldState {
    const ROWS: i8 = 3;
    const COLUMNS: i8 = 4;
    const GOAL_CELL: Position = Position { r: 0, c: 3 };
    const PENALTY_CELL: Position = Position { r: 1, c: 3 };
    const BLOCKED_CELL: Position = Position { r: 1, c: 1 };

    pub fn new() -> Self {
        return GridWorldState {
            current_position: Position { r: 0, c: 0 },
        };
    }
}

impl State for GridWorldState {
    fn get_current_position(&self) -> Position {
        return self.current_position;
    }

    fn update_current_position(&mut self, new_position: Position) {
        if new_position == Self::BLOCKED_CELL {
            return;
        }
        if new_position.r < 0 || new_position.r >= Self::ROWS {
            return;
        }
        if new_position.c < 0 || new_position.c >= Self::COLUMNS {
            return;
        }

        self.current_position = new_position;
    }

    fn evaluate(&self) -> Reward {
        return match self.current_position {
            Self::GOAL_CELL => 1.0,
            Self::PENALTY_CELL => -1.0,
            _ => 0.0_f64,
        };
    }

    fn get_legal_actions(&self) -> Vec<Box<dyn Action>> {
        let current_position = self.get_current_position();
        let actions = vec![
            Box::new(GridWorldAction::Up),
            Box::new(GridWorldAction::Down),
            Box::new(GridWorldAction::Left),
            Box::new(GridWorldAction::Right),
        ];

        let mut legal_actions: Vec<Box<dyn Action>> = vec![];
        for action in actions {
            let delta = action.delta();
            let new_position = current_position.add(delta);
            if new_position == Self::BLOCKED_CELL {
                continue;
            }
            if new_position.r < 0 || new_position.r >= Self::ROWS {
                continue;
            }
            if new_position.c < 0 || new_position.c >= Self::COLUMNS {
                continue;
            }

            legal_actions.push(action);
        }

        return legal_actions;
    }

    fn is_game_ended(&self) -> bool {
        return self.current_position == Self::GOAL_CELL
            || self.current_position == Self::PENALTY_CELL;
    }
}

fn main() {
    // Create a new state for the grid world
    let mut state: Box<dyn State> = Box::new(GridWorldState::new());
    println!("Initial position: {:#?}", state.get_current_position());

    // Get grid dimensions for reference (uses ROWS and COLUMNS constants)
    let rows = GridWorldState::ROWS;
    let columns = GridWorldState::COLUMNS;
    println!("Grid size: {rows}x{columns}");

    // Convert position to usize indices (uses to_usize method)
    let (row_idx, col_idx) = state.get_current_position().to_usize();
    println!("Position as indices: ({row_idx}, {col_idx})");

    let actions = state.get_legal_actions();
    println!("Legal actions available: {}", actions.len());

    // Apply first action to demonstrate the action application
    if let Some(action) = actions.first() {
        let action_name = action.get_name();
        action.apply_to(state.as_mut());
        println!(
            "After applying '{}' action: {:#?}",
            action_name,
            state.get_current_position()
        );
    }

    // Check if game has ended and evaluate the state
    // Uses is_game_ended() and evaluate() methods, which return Reward type
    let ended = state.is_game_ended();
    let reward: Reward = state.evaluate();
    println!("Game ended: {ended}, Reward: {reward}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_action_up_delta() {
        let action = GridWorldAction::Up;
        assert_eq!(action.delta(), Position { r: -1, c: 0 });
    }

    #[test]
    fn test_grid_action_down_delta() {
        let action = GridWorldAction::Down;
        assert_eq!(action.delta(), Position { r: 1, c: 0 });
    }

    #[test]
    fn test_grid_action_left_delta() {
        let action = GridWorldAction::Left;
        assert_eq!(action.delta(), Position { r: 0, c: -1 });
    }

    #[test]
    fn test_grid_action_right_delta() {
        let action = GridWorldAction::Right;
        assert_eq!(action.delta(), Position { r: 0, c: 1 });
    }

    #[test]
    fn test_grid_world_state_get_current_position() {
        let state = GridWorldState::new();
        let current_position = state.get_current_position();
        assert_eq!(current_position, Position{r: 0, c: 0})
    }

    #[test]
    fn test_grid_world_state_get_legal_actions() {
        let state = GridWorldState::new();
        let actions = state.get_legal_actions();
        let actions: Vec<&str> = actions.iter().map(|action| action.as_ref().get_name()).collect();
        assert_eq!(actions.len(), 2);
        assert_eq!(actions, vec!["Down", "Right"])
    }
}
