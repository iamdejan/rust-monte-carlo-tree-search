use crate::policy::RolloutPolicy;
use crate::reward::Reward;
use crate::{action::Action, state::State};

type NodeId = i8;

struct Node {
    parent: Option<Box<Node>>,
    children: Vec<Box<Node>>,
    state: Box<dyn State>,
    untried_actions: Vec<Box<dyn Action>>,
    causing_action: Option<Box<dyn Action>>,
    q: f64,
    n: f64,
}

impl Node {
    fn new(
        parent: Option<Box<Node>>,
        state: Box<dyn State>,
        causing_action: Option<Box<dyn Action>>,
    ) -> Self {
        let actions = state.get_legal_actions();
        return Node {
            parent,
            children: vec![],
            state,
            causing_action,
            untried_actions: actions,
            q: 0.0,
            n: 0.0,
        };
    }

    fn uct_best_child(&mut self, exploration_constant: f64) -> usize {
        let mut chosen_index = 0;
        let mut max_value = f64::MIN;

        let mut i = 0;
        for child in &self.children {
            let ucb1 =
                (child.q / child.n) + exploration_constant * (f64::from(self.n) / child.n).sqrt();
            if ucb1 > max_value {
                max_value = ucb1;
                chosen_index = i;
            }
            i += 1;
        }

        return chosen_index;
    }

    fn tree_policy(&mut self) -> Option<Self> {
        todo!()
    }

    fn rollout(&mut self, _policy: RolloutPolicy) -> Reward {
        todo!();
    }

    fn backpropagate(&mut self, _reward: Reward) {
        todo!()
    }
}

pub fn search(
    state: Box<dyn State>,
    policy: RolloutPolicy,
    num_of_simulations: i64,
) -> Box<dyn Action> {
    let mut root = Node::new(None, state, None);
    for _ in 0..num_of_simulations {
        let leaf_optional = root.tree_policy();
        if let Some(mut leaf) = leaf_optional {
            let reward: Reward = leaf.rollout(policy);
            leaf.backpropagate(reward);
        }
    }

    let chosen_index = root.uct_best_child(0.0);
    let chosen_child = root
        .children
        .get_mut(chosen_index)
        .expect("root.uct_best_child(0.0) should return valid index");
    return chosen_child
        .causing_action
        .take() // why using take: https://stackoverflow.com/a/57862198
        .expect("causing_action should not be None");
}
