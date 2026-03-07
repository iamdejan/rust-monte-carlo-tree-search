mod action;
mod grid_world;
mod mcts;
mod policy;
mod position;
mod reward;
mod state;

use crate::{grid_world::GridWorldState, state::State};

const MAX_STEPS: i8 = 50;

fn main() {
    let mut state: Box<dyn State> = Box::new(GridWorldState::new());
    let mut steps = 0;
    while steps < MAX_STEPS {
        if state.is_game_ended() {
            println!(
                "Game is ended! Step: {}, state={:#?}",
                steps,
                state.get_current_position()
            );
            break;
        }

        let chosen_action = mcts::search(state.clone(), policy::default, 1000);
        println!("{}", chosen_action.get_name());

        let new_state: Box<dyn State> = chosen_action.apply_to(state.as_ref());
        println!(
            "New state's position: {:#?}",
            new_state.get_current_position()
        );
        state = new_state;

        steps += 1;
    }

    // Check if we hit the step limit without reaching terminal
    if steps >= MAX_STEPS {
        println!("Reached max steps without reaching terminal");
    }
}
