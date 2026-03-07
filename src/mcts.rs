use crate::policy::RolloutPolicy;
use crate::reward::Reward;
use crate::{action::Action, state::State};

const MAX_ROLLOUT_DEPTH: i32 = 50;

/// MCTS Tree structure that manages nodes and provides index-based access
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn new(root_state: Box<dyn State>) -> Self {
        let mut root = Node {
            parent_index: None,
            children: vec![],
            state: root_state,
            untried_actions: vec![],
            causing_action: None,
            q: 0.0,
            n: 0.0,
        };

        // Populate root's untried actions
        root.untried_actions = root.state.get_legal_actions();

        let tree = Tree { nodes: vec![root] };
        return tree;
    }

    /// Get a mutable reference to a node by index
    fn get_node_mut(&mut self, index: usize) -> Option<&mut Node> {
        return self.nodes.get_mut(index);
    }

    /// Get a reference to a node by index
    fn get_node(&self, index: usize) -> Option<&Node> {
        return self.nodes.get(index);
    }

    /// Add a new child node and return its index
    fn add_child(
        &mut self,
        parent_index: usize,
        state: Box<dyn State>,
        action: Box<dyn Action>,
    ) -> usize {
        let child_index = self.nodes.len();

        let mut child = Node {
            parent_index: Some(parent_index),
            children: vec![],
            state,
            untried_actions: vec![],
            causing_action: Some(action),
            q: 0.0,
            n: 0.0,
        };

        // Populate child's untried actions
        child.untried_actions = child.state.get_legal_actions();

        self.nodes.push(child);

        // Add child to parent's children list
        self.nodes[parent_index].children.push(child_index);

        return child_index;
    }
}

/// MCTS Node that uses indices for parent references instead of direct pointers
/// to avoid Rust ownership issues with cloned parent pointers.
#[derive(Clone)]
struct Node {
    /// Index of parent node in the tree's nodes vector (None for root)
    parent_index: Option<usize>,
    /// Indices of child nodes in the tree's nodes vector
    children: Vec<usize>,
    /// The state at this node
    state: Box<dyn State>,
    /// Actions that haven't been tried from this node yet
    untried_actions: Vec<Box<dyn Action>>,
    /// The action that led to this node from its parent
    causing_action: Option<Box<dyn Action>>,
    /// Total reward accumulated through this node
    q: f64,
    /// Number of times this node has been visited
    n: f64,
}

impl Node {
    fn is_terminal(&self) -> bool {
        return self.state.is_game_ended();
    }

    fn is_fully_expanded(&self) -> bool {
        return self.untried_actions.is_empty();
    }

    fn remove_first_untried_action(&mut self) -> Box<dyn Action> {
        return self.untried_actions.remove(0);
    }

    /// UCB1-based selection that uses tree's node access
    fn uct_best_child(tree: &Tree, node_index: usize, exploration_constant: f64) -> usize {
        let node = tree.get_node(node_index).unwrap();

        // If there are no children, return 0 (shouldn't happen in normal operation)
        if node.children.is_empty() {
            return 0;
        }

        let mut chosen_index = node.children[0];
        let mut max_value = f64::MIN;

        for &child_index in &node.children {
            let child = tree.get_node(child_index).unwrap();

            // Handle unvisited children - give them infinite value to ensure exploration
            if child.n == 0.0 {
                return child_index;
            }

            let ucb1 = (child.q / child.n) + exploration_constant * (node.n.ln() / child.n).sqrt();
            if ucb1 > max_value {
                max_value = ucb1;
                chosen_index = child_index;
            }
        }

        return chosen_index;
    }

    /// Tree policy that traverses the tree using index-based access
    fn tree_policy(tree: &mut Tree, node_index: usize) -> Option<usize> {
        let mut current_index = node_index;

        loop {
            let current = tree.get_node(current_index).unwrap();

            if current.is_terminal() {
                return Some(current_index);
            }

            if !current.is_fully_expanded() {
                // Expand this node
                let action = tree
                    .get_node_mut(current_index)
                    .unwrap()
                    .remove_first_untried_action();

                let current_state = tree.get_node(current_index).unwrap().state.clone();
                let new_state = action.apply_to(current_state.as_ref());

                let new_index = tree.add_child(current_index, new_state, action);
                return Some(new_index);
            }

            // All actions tried, select best child
            let chosen_index = Self::uct_best_child(tree, current_index, 1.4);
            current_index = chosen_index;
        }
    }

    /// Rollout simulation using index-based access
    fn rollout(tree: &Tree, node_index: usize, policy: RolloutPolicy) -> Reward {
        let mut current_state: Box<dyn State> = tree.get_node(node_index).unwrap().state.clone();
        let mut depth: i32 = 0;
        let gamma: f64 = 0.95;

        while depth < MAX_ROLLOUT_DEPTH && !current_state.is_game_ended() {
            let action_option = policy(current_state.as_mut());
            if let Some(action) = action_option {
                current_state = action.apply_to(current_state.as_ref());
                depth += 1;
            } else {
                break;
            }
        }

        let reward = current_state.evaluate();
        return reward * gamma.powi(depth);
    }

    /// Backpropagation using index-based access
    fn backpropagate(tree: &mut Tree, node_index: usize, reward: Reward) {
        let mut current_index = node_index;

        loop {
            {
                let current = tree.get_node_mut(current_index).unwrap();
                current.q += reward;
                current.n += 1.0;
            }

            match tree.get_node(current_index).unwrap().parent_index {
                Some(parent_index) => {
                    current_index = parent_index;
                }
                None => break, // Reached root
            }
        }
    }
}

pub fn search(
    state: Box<dyn State>,
    policy: RolloutPolicy,
    num_of_simulations: i64,
) -> Box<dyn Action> {
    let mut tree = Tree::new(state);
    let root_index = 0;

    for _ in 0..num_of_simulations {
        if let Some(leaf_index) = Node::tree_policy(&mut tree, root_index) {
            let reward: Reward = Node::rollout(&tree, leaf_index, policy);
            Node::backpropagate(&mut tree, leaf_index, reward);
        }
    }

    let chosen_index = Node::uct_best_child(&tree, root_index, 0.0);
    let chosen_child = tree.get_node(chosen_index).unwrap();

    return chosen_child
        .causing_action
        .as_ref()
        .expect("causing_action should not be None")
        .clone_box();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_world::GridWorldState;

    /// Tests for the MCTS Node structure
    mod node_tests {
        use super::*;

        /// Tests that Node::new creates a node with correct initial values.
        #[test]
        fn test_node_new_initializes_correctly() {
            let state = Box::new(GridWorldState::new());
            let tree = Tree::new(state);
            let root = tree.get_node(0).unwrap();

            assert_eq!(root.q, 0.0);
            assert_eq!(root.n, 0.0);
            assert!(root.parent_index.is_none());
            assert!(root.children.is_empty());
        }

        /// Tests that root node populates untried_actions from the state's legal actions.
        #[test]
        fn test_node_new_populates_untried_actions() {
            let state = Box::new(GridWorldState::new());
            let tree = Tree::new(state);
            let root = tree.get_node(0).unwrap();

            // At origin (0, 0), there are 2 legal actions: Down and Right
            assert_eq!(root.untried_actions.len(), 2);
        }

        /// Tests that is_terminal correctly identifies terminal states.
        #[test]
        fn test_is_terminal_identifies_terminal_states() {
            let state = Box::new(GridWorldState::new());
            let tree = Tree::new(state);
            let root = tree.get_node(0).unwrap();

            assert!(!root.is_terminal());
        }

        /// Tests that is_fully_expanded correctly identifies when all actions are tried.
        #[test]
        fn test_is_fully_expanded() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);
            let root = tree.get_node_mut(0).unwrap();

            // Initially should not be fully expanded
            assert!(!root.is_fully_expanded());

            // Remove all untried actions
            root.untried_actions.clear();
            assert!(root.is_fully_expanded());
        }
    }

    /// Tests for UCT (Upper Confidence Bound for Trees) selection
    mod uct_tests {
        use super::*;

        /// Tests that uct_best_child handles unvisited children correctly.
        #[test]
        fn test_uct_best_child_unvisited() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            // Add two children
            let child1_state = Box::new(GridWorldState::new());
            tree.add_child(
                0,
                child1_state,
                Box::new(crate::grid_world::GridWorldAction::Down),
            );

            let child2_state = Box::new(GridWorldState::new());
            tree.add_child(
                0,
                child2_state,
                Box::new(crate::grid_world::GridWorldAction::Right),
            );

            // First call should return first unvisited child
            let chosen = Node::uct_best_child(&tree, 0, 1.4);
            assert_eq!(chosen, 1); // First child (index 1 in tree)
        }
    }

    /// Integration tests for the MCTS search function
    mod search_tests {
        use super::*;
        use crate::policy;

        /// Tests that search returns a valid action.
        #[test]
        fn test_search_returns_action() {
            let state = Box::new(GridWorldState::new());
            let action = search(state, policy::default, 10);

            // Verify the action name is valid (Down or Right from origin)
            let name = action.get_name();
            assert!(
                name == "Down" || name == "Right",
                "Expected Down or Right, got {}",
                name
            );
        }

        /// Tests that search with different simulation counts returns valid actions.
        #[test]
        fn test_search_with_different_simulation_counts() {
            // With few simulations
            let state1 = Box::new(GridWorldState::new());
            let action_few = search(state1, policy::default, 5);
            let few_name = action_few.get_name();

            // With more simulations
            let state2 = Box::new(GridWorldState::new());
            let action_many = search(state2, policy::default, 20);
            let many_name = action_many.get_name();

            // Both should return valid action names (Down or Right from origin)
            assert!(
                few_name == "Down" || few_name == "Right",
                "Expected Down or Right, got {}",
                few_name
            );
            assert!(
                many_name == "Down" || many_name == "Right",
                "Expected Down or Right, got {}",
                many_name
            );
        }
    }
}
