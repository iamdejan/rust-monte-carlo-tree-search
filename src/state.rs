use crate::{action::Action, position::Position, reward::Reward};

pub trait State {
    fn get_current_position(&self) -> Position;
    fn update_current_position(&mut self, new_position: Position);
    fn evaluate(&self) -> Reward;
    fn get_legal_actions(&self) -> Vec<Box<dyn Action>>;
    fn is_game_ended(&self) -> bool;
}