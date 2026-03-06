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

    let new_state = chosen_action.apply_to(initial_state.as_ref());
    println!(
        "New state's position: {:#?}",
        new_state.get_current_position()
    );
    println!("Goal state: {:#?}", GridWorldState::GOAL_CELL);
    println!("Penalty state: {:#?}", GridWorldState::PENALTY_CELL);
}
