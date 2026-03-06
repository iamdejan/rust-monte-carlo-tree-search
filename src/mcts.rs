use crate::policy::RolloutPolicy;
use crate::reward::Reward;
use crate::{action::Action, state::State};

type NodeId = i8;

struct Node {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    state: Box<dyn State>,
    untried_actions: Vec<Box<dyn Action>>,
    causing_action: Option<Box<dyn Action>>,
    q: f64,
    n: f64,
}

impl Node {
    fn new(
        parent: Option<NodeId>,
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

    fn uct_best_child(&mut self, _c: f64) -> Self {
        todo!()
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
    let mut leaf_optional: Option<Node> = None;
    for _ in 0..num_of_simulations {
        leaf_optional = root.tree_policy();
        if let Some(mut leaf) = leaf_optional {
            let reward: Reward = leaf.rollout(policy);
            leaf.backpropagate(reward);
        }
    }

    return root
        .uct_best_child(0.0)
        .causing_action
        .expect("root.uct_best_child(0.0) should have causing_action");
}
