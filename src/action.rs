use crate::state::State;

pub trait Action {
    fn apply_to(&self, state: &mut dyn State);
    fn get_name(&self) -> &'static str;
}