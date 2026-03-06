use crate::policy::RolloutPolicy;
use crate::reward::Reward;
use crate::{action::Action, state::State};

#[derive(Clone)]
struct Node {
    parent: Option<Box<Node>>,
    children: Vec<Node>,
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

        for (i, child) in self.children.iter().enumerate() {
            let ucb1 = (child.q / child.n) + exploration_constant * (self.n / child.n).sqrt();
            if ucb1 > max_value {
                max_value = ucb1;
                chosen_index = i;
            }
        }

        return chosen_index;
    }

    fn tree_policy(mut self: &mut Self) -> Option<&mut Self> {
        while !self.is_terminal() {
            if !self.is_fully_expanded() {
                return self.expand();
            }

            let chosen_index = self.uct_best_child(1.4);
            self = self.children.get_mut(chosen_index)?;
        }

        return Some(self);
    }

    fn rollout(&mut self, policy: RolloutPolicy) -> Reward {
        let mut current_state: Box<dyn State> = self.state.clone();
        while !current_state.is_game_ended() {
            let action_option = policy(current_state.as_mut());
            if let Some(action) = action_option {
                current_state = action.apply_to(&current_state);
            }
        }

        let reward = current_state.evaluate();
        return reward;
    }

    fn backpropagate(&mut self, reward: Reward) {
        let mut current = self;
        while !current.is_root() {
            current.q += reward;
            current.n += 1.0;
            current = current.parent.as_mut().unwrap();
        }
        current.n += 1.0;
    }

    fn is_terminal(&self) -> bool {
        todo!();
    }

    fn is_fully_expanded(&self) -> bool {
        return self.untried_actions.is_empty();
    }

    fn remove_first_untried_action(&mut self) -> Option<Box<dyn Action>> {
        return Some(self.untried_actions.remove(0));
    }

    fn expand(&mut self) -> Option<&mut Self> {
        let action_option = self.remove_first_untried_action();
        if action_option.is_none() {
            return Option::None;
        }

        let current = self.clone();
        let action = action_option.unwrap();
        let new_state = action.apply_to(&self.state);
        let expanded_child = Node::new(
            Option::Some(Box::new(current)),
            new_state,
            Option::Some(action),
        );
        self.children.push(expanded_child);
        return self.children.last_mut();
    }

    fn is_root(&self) -> bool {
        return self.parent.is_none();
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
        if let Some(leaf) = leaf_optional {
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
