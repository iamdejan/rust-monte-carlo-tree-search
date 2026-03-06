use crate::state::State;

pub trait Action {
    fn apply_to(&self, state: &Box<dyn State>) -> Box<dyn State>;
    fn get_name(&self) -> &'static str;
}
