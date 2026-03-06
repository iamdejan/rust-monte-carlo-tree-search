mod action;
mod grid_world;
mod mcts;
mod policy;
mod position;
mod reward;
mod state;

use crate::{grid_world::GridWorldState, state::State};

fn main() {
    let initial_state: Box<dyn State> = Box::new(GridWorldState::new());
    let chosen_action = mcts::search(initial_state.clone(), policy::default, 100);
    println!("{}", chosen_action.get_name());

    let new_state = chosen_action.apply_to(&initial_state);
    println!(
        "New state's position: {:#?}",
        new_state.get_current_position()
    );
    println!("Goal state: {:#?}", GridWorldState::GOAL_CELL);
    println!("Penalty state: {:#?}", GridWorldState::PENALTY_CELL);
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
            let new_state = action.as_ref().apply_to(&state);
            assert_eq!(new_state.get_current_position(), Position { r: 1, c: 0 });
        } else {
            panic!("action should exist");
        }
    }
}
