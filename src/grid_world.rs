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
    use crate::grid_world::GridWorldAction;
    use crate::position::Position;
    use crate::state::State;

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
        assert_eq!(current_position, Position { r: 0, c: 0 });
    }

    #[test]
    fn test_grid_world_state_get_legal_actions() {
        let state = GridWorldState::new();
        let actions = state.get_legal_actions();
        let actions: Vec<&str> = actions
            .iter()
            .map(|action| action.as_ref().get_name())
            .collect();
        assert_eq!(actions.len(), 2);
        assert_eq!(actions, vec!["Down", "Right"]);
    }

    #[test]
    fn test_grid_world_state_apply_action() {
        let state: Box<dyn State> = Box::new(GridWorldState::new());
        let actions = state.get_legal_actions();

        if let Some(action) = actions.first() {
            let new_state = action.as_ref().apply_to(state.as_ref());
            assert_eq!(new_state.get_current_position(), Position { r: 1, c: 0 });
        } else {
            panic!("action should exist");
        }
    }
}
