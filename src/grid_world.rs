use crate::action::Action;
use crate::position::Position;
use crate::reward::Reward;
use crate::state::State;

#[derive(Debug, Clone)]
pub enum GridWorldAction {
    Up,
    Down,
    Left,
    Right,
}

impl GridWorldAction {
    pub const fn delta(&self) -> Position {
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
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Up => return "Up",
            Self::Down => return "Down",
            Self::Left => return "Left",
            Self::Right => return "Right",
        }
    }
}

impl Action for GridWorldAction {
    fn apply_to(&self, state: &dyn State) -> Box<dyn State> {
        let current_position = state.get_current_position();
        let delta = self.delta();
        let new_position = current_position.add(delta);

        let mut new_state = GridWorldState::new();
        new_state.update_current_position(new_position);
        return Box::new(new_state);
    }

    fn get_name(&self) -> &'static str {
        return self.name();
    }

    fn clone_box(&self) -> Box<dyn Action> {
        return Box::new(self.clone());
    }
}

#[derive(Debug, Clone)]
pub struct GridWorldState {
    pub current_position: Position,
}

impl GridWorldState {
    pub const ROWS: i8 = 3;
    pub const COLUMNS: i8 = 4;
    pub const GOAL_CELL: Position = Position { r: 0, c: 3 };
    pub const PENALTY_CELL: Position = Position { r: 1, c: 3 };
    pub const BLOCKED_CELL: Position = Position { r: 1, c: 1 };

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

    fn clone_box(&self) -> Box<dyn State> {
        return Box::new(self.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests for GridWorldAction
    mod grid_world_action_tests {
        use super::*;

        /// Tests that delta() returns correct position delta for each action direction.
        ///
        /// # Steps
        /// 1. Create each GridWorldAction variant
        /// 2. Call delta() on each action
        /// 3. Verify the returned Position matches expected delta values
        ///
        /// # Expected Results
        /// - Up: (-1, 0), Down: (1, 0), Left: (0, -1), Right: (0, 1)
        #[test]
        fn test_delta_returns_correct_position_deltas() {
            assert_eq!(GridWorldAction::Up.delta(), Position { r: -1, c: 0 });
            assert_eq!(GridWorldAction::Down.delta(), Position { r: 1, c: 0 });
            assert_eq!(GridWorldAction::Left.delta(), Position { r: 0, c: -1 });
            assert_eq!(GridWorldAction::Right.delta(), Position { r: 0, c: 1 });
        }

        /// Tests that name() returns the correct string representation for each action.
        ///
        /// # Steps
        /// 1. Create each GridWorldAction variant
        /// 2. Call name() on each action
        /// 3. Verify the returned string matches expected names
        ///
        /// # Expected Results
        /// - Up returns "Up", Down returns "Down", etc.
        #[test]
        fn test_name_returns_correct_action_names() {
            assert_eq!(GridWorldAction::Up.name(), "Up");
            assert_eq!(GridWorldAction::Down.name(), "Down");
            assert_eq!(GridWorldAction::Left.name(), "Left");
            assert_eq!(GridWorldAction::Right.name(), "Right");
        }
    }

    /// Tests for GridWorldState
    mod grid_world_state_tests {
        use super::*;

        /// Tests that new() creates a state with the correct initial position (0, 0).
        ///
        /// # Steps
        /// 1. Create a new GridWorldState using new()
        /// 2. Verify the current_position is at origin (0, 0)
        ///
        /// # Expected Results
        /// - Initial position is {r: 0, c: 0}
        #[test]
        fn test_new_creates_state_at_origin() {
            let state = GridWorldState::new();
            assert_eq!(state.current_position, Position { r: 0, c: 0 });
        }

        /// Tests that evaluate() returns correct rewards for different cell types.
        ///
        /// # Steps
        /// 1. Create states at goal cell (0, 3), penalty cell (1, 3), and regular cell
        /// 2. Call evaluate() on each state
        /// 3. Verify the returned rewards are correct
        ///
        /// # Expected Results
        /// - Goal cell: 1.0, Penalty cell: -1.0, Regular cell: 0.0
        #[test]
        fn test_evaluate_returns_correct_rewards() {
            // Test goal cell reward
            let mut goal_state = GridWorldState::new();
            goal_state.update_current_position(GridWorldState::GOAL_CELL);
            assert_eq!(goal_state.evaluate(), 1.0);

            // Test penalty cell reward
            let mut penalty_state = GridWorldState::new();
            penalty_state.update_current_position(GridWorldState::PENALTY_CELL);
            assert_eq!(penalty_state.evaluate(), -1.0);

            // Test regular cell reward (should be 0.0)
            let regular_state = GridWorldState::new();
            assert_eq!(regular_state.evaluate(), 0.0);
        }

        /// Tests that is_game_ended() correctly identifies terminal states.
        ///
        /// # Steps
        /// 1. Create states at goal cell, penalty cell, and regular cell
        /// 2. Call is_game_ended() on each state
        /// 3. Verify terminal states return true, non-terminal returns false
        ///
        /// # Expected Results
        /// - Goal and penalty cells return true, regular cells return false
        #[test]
        fn test_is_game_ended_identifies_terminal_states() {
            // At goal cell - should be terminal
            let mut goal_state = GridWorldState::new();
            goal_state.update_current_position(GridWorldState::GOAL_CELL);
            assert!(goal_state.is_game_ended());

            // At penalty cell - should be terminal
            let mut penalty_state = GridWorldState::new();
            penalty_state.update_current_position(GridWorldState::PENALTY_CELL);
            assert!(penalty_state.is_game_ended());

            // At regular cell - should not be terminal
            let regular_state = GridWorldState::new();
            assert!(!regular_state.is_game_ended());
        }

        /// Tests that update_current_position correctly handles boundaries and blocked cells.
        ///
        /// # Steps
        /// 1. Create a state and try to update position to blocked cell, outside bounds
        /// 2. Verify these invalid positions are rejected
        /// 3. Test valid position updates work correctly
        ///
        /// # Expected Results
        /// - Blocked cell updates are rejected, position remains unchanged
        /// - Out of bounds updates are rejected, position remains unchanged
        /// - Valid updates succeed
        #[test]
        fn test_update_current_position_handles_boundaries() {
            let mut state = GridWorldState::new();

            // Test blocked cell is rejected
            state.update_current_position(GridWorldState::BLOCKED_CELL);
            assert_eq!(state.current_position, Position { r: 0, c: 0 });

            // Test out of bounds (negative row) is rejected
            state.update_current_position(Position { r: -1, c: 0 });
            assert_eq!(state.current_position, Position { r: 0, c: 0 });

            // Test out of bounds (negative column) is rejected
            state.update_current_position(Position { r: 0, c: -1 });
            assert_eq!(state.current_position, Position { r: 0, c: 0 });

            // Test out of bounds (row >= ROWS) is rejected
            state.update_current_position(Position {
                r: GridWorldState::ROWS,
                c: 0,
            });
            assert_eq!(state.current_position, Position { r: 0, c: 0 });

            // Test out of bounds (col >= COLUMNS) is rejected
            state.update_current_position(Position {
                r: 0,
                c: GridWorldState::COLUMNS,
            });
            assert_eq!(state.current_position, Position { r: 0, c: 0 });

            // Test valid position update works
            state.update_current_position(Position { r: 1, c: 0 });
            assert_eq!(state.current_position, Position { r: 1, c: 0 });
        }

        /// Tests that get_legal_actions() returns correct actions from different positions.
        ///
        /// # Steps
        /// 1. Create states at origin, near walls, and near blocked cell
        /// 2. Call get_legal_actions() on each state
        /// 3. Verify correct number and types of legal actions are returned
        ///
        /// # Expected Results
        /// - Origin: 2 legal actions (Down, Right)
        /// - Near blocked cell: should exclude actions leading to blocked cell
        #[test]
        fn test_get_legal_actions_returns_correct_actions() {
            // At origin (0, 0): Up and Left would go out of bounds, Down and Right are valid
            let state = GridWorldState::new();
            let legal_actions = state.get_legal_actions();
            assert_eq!(legal_actions.len(), 2);

            // Verify the specific actions available at origin
            let action_names: Vec<&str> = legal_actions.iter().map(|a| a.get_name()).collect();
            assert!(action_names.contains(&"Down"));
            assert!(action_names.contains(&"Right"));
            assert!(!action_names.contains(&"Up"));
            assert!(!action_names.contains(&"Left"));
        }

        /// Tests that get_legal_actions() handles blocked cell correctly.
        ///
        /// # Steps
        /// 1. Create state at position (0, 1) which is adjacent to blocked cell (1, 1)
        /// 2. Call get_legal_actions()
        /// 3. Verify Down action is excluded (would lead to blocked cell)
        ///
        /// # Expected Results
        /// - Down action should not be available when adjacent to blocked cell
        #[test]
        fn test_get_legal_actions_excludes_blocked_cell() {
            let mut state = GridWorldState::new();
            state.update_current_position(Position { r: 0, c: 1 });

            let legal_actions = state.get_legal_actions();
            let action_names: Vec<&str> = legal_actions.iter().map(|a| a.get_name()).collect();

            // Down would lead to blocked cell (1, 1), so it should not be available
            assert!(!action_names.contains(&"Down"));
        }

        /// Tests that Action::apply_to correctly moves the agent to new position.
        ///
        /// # Steps
        /// 1. Create a GridWorldState and a GridWorldAction (Right)
        /// 2. Apply the action to the state
        /// 3. Verify the resulting state has the correct new position
        ///
        /// # Expected Results
        /// - Right action from (0, 0) moves to (0, 1)
        #[test]
        fn test_apply_to_moves_agent_correctly() {
            let state = GridWorldState::new();
            let action = GridWorldAction::Right;

            let new_state = action.apply_to(&state);
            assert_eq!(new_state.get_current_position(), Position { r: 0, c: 1 });
        }

        /// Tests that apply_to handles wall collisions correctly.
        ///
        /// # Steps
        /// 1. Create a state at position (0, 0) and try to move Up (which is out of bounds)
        /// 2. Apply the action
        /// 3. Verify position doesn't change (or verify valid move only)
        ///
        /// # Expected Results
        /// - Valid moves only change position when within bounds and not blocked
        #[test]
        fn test_apply_to_handles_wall_collisions() {
            let state = GridWorldState::new();

            // Try moving up from origin - this should result in staying at origin
            // because the grid world implementation rejects invalid moves
            let action = GridWorldAction::Up;
            let new_state = action.apply_to(&state);

            // The new state will have position (0, 0) because update_current_position
            // rejects the out-of-bounds update
            assert_eq!(new_state.get_current_position(), Position { r: 0, c: 0 });
        }
    }
}
