use crate::state::State;

pub trait Action {
    fn apply_to(&self, state: &Box<dyn State>) -> Box<dyn State>;
    fn get_name(&self) -> &'static str;
    fn clone_box(&self) -> Box<dyn Action>;
}

impl Clone for Box<dyn Action> {
    fn clone(&self) -> Self {
        return self.clone_box();
    }
}
