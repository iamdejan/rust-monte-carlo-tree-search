//! Monte Carlo Tree Search (MCTS) implementation for decision-making in stochastic environments.
//!
//! This module provides an implementation of the MCTS algorithm, which is used to find optimal
//! decisions in environments where the outcomes are uncertain. MCTS combines the precision of
//! tree search with the generality of random sampling.
//!
//! ## Algorithm Overview
//!
//! The MCTS algorithm consists of four phases that are repeated for a fixed number of simulations:
//! 1. **Selection**: Starting from the root, select child nodes using UCB1 until reaching a node
//!    that is not fully expanded or is a terminal state.
//! 2. **Expansion**: Add one or more child nodes to the selected node.
//! 3. **Simulation (Rollout)**: Perform a random playout from the new node until reaching a terminal
//!    state or maximum depth.
//! 4. **Backpropagation**: Update the statistics (visit count and total reward) for all nodes
//!    along the path from the expanded node back to the root.
//!
//! ## Why Index-Based Tree Structure?
//!
//! This implementation uses indices instead of direct pointers to reference nodes in the tree.
//! This design choice avoids Rust's ownership issues with self-referential structures while
//! maintaining efficient O(1) access to any node by its index.

use crate::policy::RolloutPolicy;
use crate::reward::Reward;
use crate::{action::Action, state::State};

/// Maximum depth for the rollout/simulation phase.
/// This prevents infinite loops in states where the game doesn't terminate
/// and limits computation time per simulation.
const MAX_ROLLOUT_DEPTH: i32 = 50;

/// Tree structure that manages nodes and provides index-based access.
///
/// Using indices instead of direct references allows us to store parent pointers
/// without lifetime complexity. Each node knows its parent and children by their
/// indices in the nodes vector.
struct Tree {
    /// Vector storing all nodes in the tree. Index serves as the node's unique identifier.
    nodes: Vec<Node>,
}

impl Tree {
    /// Creates a new MCTS tree with a root node initialized from the given state.
    ///
    /// The root node represents the current game state from which we want to find
    /// the best action. We populate its `untried_actions` with all legal actions available
    /// from this state, which drives the expansion phase.
    ///
    /// # Arguments
    /// * `root_state` - The initial state to create the tree from
    ///
    /// # Returns
    /// A new Tree with a single root node
    fn new(root_state: Box<dyn State>) -> Self {
        // Create the root node with no parent (it's the starting point)
        let mut root = Node {
            parent_index: None,
            children: vec![],
            state: root_state,
            untried_actions: vec![],
            // Action that led to this node - None for root since it represents initial state
            causing_action: None,
            // Q-value: cumulative reward through this node (starts at 0)
            q: 0.0,
            // Visit count: number of times this node has been part of a simulation
            n: 0.0,
        };

        // Populate root's untried actions by querying the state for all legal moves.
        // This is crucial for the expansion phase - we need to know what actions
        // are available to try them one by one.
        root.untried_actions = root.state.get_legal_actions();

        let tree = Tree { nodes: vec![root] };
        return tree;
    }

    /// Get a mutable reference to a node by its index.
    ///
    /// Returns None if the index is out of bounds.
    fn get_node_mut(&mut self, index: usize) -> Option<&mut Node> {
        return self.nodes.get_mut(index);
    }

    /// Get a reference to a node by its index.
    ///
    /// Returns None if the index is out of bounds.
    fn get_node(&self, index: usize) -> Option<&Node> {
        return self.nodes.get(index);
    }

    /// Add a new child node to an existing parent node.
    ///
    /// This is called during the expansion phase. We:
    /// 1. Create a new node with the new state resulting from applying the action
    /// 2. Add it to our nodes vector (getting a new unique index)
    /// 3. Register this new node as a child of the parent
    ///
    /// # Arguments
    /// * `parent_index` - Index of the parent node in the tree
    /// * `state` - The new state after applying the action
    /// * `action` - The action that led to this new state
    ///
    /// # Returns
    /// The index of the newly created child node
    fn add_child(
        &mut self,
        parent_index: usize,
        state: Box<dyn State>,
        action: Box<dyn Action>,
    ) -> usize {
        // Get the index where this new node will be placed
        // (current length becomes the new index since we use push)
        let child_index = self.nodes.len();

        // Create the child node with appropriate parent reference
        let mut child = Node {
            parent_index: Some(parent_index),
            children: vec![],
            state,
            untried_actions: vec![],
            // Record which action led to this node for later retrieval
            causing_action: Some(action),
            q: 0.0,
            n: 0.0,
        };

        // Populate child's untried actions - this is essential because we need
        // to know what actions are available from this new state to continue
        // the tree expansion in future simulations
        child.untried_actions = child.state.get_legal_actions();

        // Add the child to our nodes collection
        self.nodes.push(child);

        // Register this child in the parent's children list so we can traverse to it
        self.nodes[parent_index].children.push(child_index);

        return child_index;
    }
}

/// MCTS Node that represents a game state in the search tree.
///
/// Each node stores:
/// - The game state at this point
/// - Statistics for the UCB1 formula (q: total reward, n: visit count)
/// - Which actions have been tried (`untried_actions`)
/// - References to parent and children via indices
///
/// Using indices instead of Rc<> or references avoids complex lifetime management
/// and makes backpropagation straightforward - we just follow `parent_index` backwards.
#[derive(Clone)]
struct Node {
    /// Index of parent node in the tree's nodes vector (None for root)
    parent_index: Option<usize>,
    /// Indices of child nodes in the tree's nodes vector
    children: Vec<usize>,
    /// The state at this node - represents the game configuration
    state: Box<dyn State>,
    /// Actions that haven't been tried from this node yet.
    /// These drive the expansion phase - we try each action exactly once
    /// before considering the node fully expanded.
    untried_actions: Vec<Box<dyn Action>>,
    /// The action that led to this node from its parent.
    /// This is needed to return the chosen action to the caller.
    causing_action: Option<Box<dyn Action>>,
    /// Total reward accumulated through this node.
    /// Used in UCB1 to estimate the value of this node.
    q: f64,
    /// Number of times this node has been visited during simulations.
    /// Higher visit count means more confidence in the q-value estimate.
    n: f64,
}

impl Node {
    /// Checks if this node represents a terminal state (game ended).
    ///
    /// Terminal states are important because:
    /// 1. We don't expand from them (no actions to try)
    /// 2. Their reward is final - no need for further simulation
    fn is_terminal(&self) -> bool {
        return self.state.is_game_ended();
    }

    /// Checks if all legal actions have been tried from this node.
    ///
    /// A node is "fully expanded" when `untried_actions` is empty, meaning
    /// we've tried all possible moves from this state. When fully expanded,
    /// the selection phase uses UCB1 to pick among children instead of expanding.
    fn is_fully_expanded(&self) -> bool {
        return self.untried_actions.is_empty();
    }

    /// Removes and returns the first untried action.
    ///
    /// This is called during expansion to get an action to try.
    /// We remove from the front to ensure fair ordering (FIFO).
    fn remove_first_untried_action(&mut self) -> Box<dyn Action> {
        return self.untried_actions.remove(0);
    }
}

/// Performs Monte Carlo Tree Search to find the best action from the given state.
///
/// This is the main entry point for the MCTS algorithm. It runs a specified number
/// of simulations, each consisting of selection/expansion, simulation, and backpropagation.
/// After all simulations, it returns the action that led to the most visited child
/// of the root (the "best" action according to MCTS).
///
/// # Arguments
/// * `state` - The current state to search from
/// * `policy` - The rollout policy to use during simulation phase
/// * `num_of_simulations` - How many MCTS iterations to run (more = better but slower)
///
/// # Returns
/// The best action to take from the current state (boxed for trait object return)
///
/// # Why return the action with most visits?
///
/// While UCB1 selects children during search based on the exploration/exploitation
/// tradeoff, at the end we choose the action with the highest visit count. This is
/// more robust because:
/// 1. Visit count is a direct measure of how often an action was successful
/// 2. It doesn't depend on the exploration constant C
/// 3. It naturally handles the exploration done during search
pub fn search(
    state: Box<dyn State>,
    policy: RolloutPolicy,
    num_of_simulations: i64,
) -> Box<dyn Action> {
    // Initialize the tree with the starting state
    let mut tree = Tree::new(state);
    let root_index = 0;

    // Run the main MCTS loop: selection -> expansion -> simulation -> backpropagation
    for _ in 0..num_of_simulations {
        // Selection + Expansion: find a node to simulate from
        let leaf_index = select_and_expand(&mut tree, root_index);

        // Simulation: play out the game randomly from this node
        let reward: Reward = rollout(&tree, leaf_index, policy);

        // Backpropagation: update all nodes along the path with the result
        backpropagate(&mut tree, leaf_index, reward);
    }

    // After all simulations, select the action with highest visit count
    // We use UCB1 with exploration_constant=0 to select purely based on exploitaton
    // (i.e., highest average reward / visit count)
    let chosen_index = uct_best_child(&tree, root_index, 0.0);
    let chosen_child = tree.get_node(chosen_index).unwrap();

    // Return the action that led to this child (cloned to transfer ownership)
    return chosen_child
        .causing_action
        .as_ref()
        .expect("causing_action should not be None")
        .clone_box();
}

/// Selects the best child node using the UCB1 (Upper Confidence Bound) formula.
///
/// The UCB1 formula balances exploration vs exploitation:
/// `UCB1 = q/n + C * sqrt(ln(N)/n)`
/// - q/n: exploitation term - average reward from this child
/// - C * sqrt(ln(N)/n): exploration term - encourages trying less-visited nodes
/// - C=1.4 is a common default that balances exploration/exploitation
///
/// Special handling: Unvisited children (n=0) are given infinite value to ensure
/// every action gets tried at least once before being compared statistically.
///
/// # Arguments
/// * `tree` - Reference to the tree containing all nodes
/// * `node_index` - Index of the current node
/// * `exploration_constant` - The C parameter in UCB1 (typically sqrt(2) or 1.4)
///
/// # Returns
/// Index of the selected child node
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

        // Handle unvisited children - give them infinite value to ensure exploration.
        // This is critical: without it, we'd never try new actions because the
        // average reward (q/n) starts at 0 and may never improve.
        if child.n == 0.0 {
            return child_index;
        }

        // UCB1 formula: exploitation + exploration
        // Using natural log (ln) as per original UCB1 paper
        let ucb1 = (child.q / child.n) + exploration_constant * (node.n.ln() / child.n).sqrt();
        if ucb1 > max_value {
            max_value = ucb1;
            chosen_index = child_index;
        }
    }

    return chosen_index;
}

/// Selection and Expansion phase combined.
///
/// Starting at the root node, MCTS moves down the tree using a selection rule
/// until it reaches a node that either:
/// 1. Is a terminal state (game ended) - return this node for simulation
/// 2. Is not fully expanded - expand by adding a new child
///
/// The most common rule is UCT (Upper Confidence Bounds for Trees), which balances:
/// - Exploitation: choosing moves with higher average reward
/// - Exploration: trying moves with less information
///
/// # Arguments
/// * `tree` - Mutable reference to the tree
/// * `node_index` - Starting node index (typically root = 0)
///
/// # Returns
/// Index of the node to use for rollout simulation
fn select_and_expand(tree: &mut Tree, node_index: usize) -> usize {
    let mut current_index = node_index;

    loop {
        let current = tree.get_node(current_index).unwrap();

        // If we've reached a terminal state, no point in expanding further
        // The reward from this state will be backpropagated
        if current.is_terminal() {
            return current_index;
        }

        // If this node has untried actions, expand by trying one of them
        // This is the "expansion" phase - we add a new branch to the tree
        if !current.is_fully_expanded() {
            return expand(tree, current_index);
        }

        // All actions have been tried - use UCB1 to select the most promising child
        // This is the "selection" phase - we're exploiting what we've learned
        let chosen_index = uct_best_child(tree, current_index, 1.4);
        current_index = chosen_index;
    }
}

/// Expansion phase: add a new child node to the tree.
///
/// Takes one untried action from the current node, applies it to the current state,
/// and adds the resulting state as a new child node. This is called during
/// `select_and_expand` when we find a node that isn't fully expanded.
///
/// # Arguments
/// * `tree` - Mutable reference to the tree
/// * `current_index` - Index of the node to expand
///
/// # Returns
/// Index of the newly created child node
fn expand(tree: &mut Tree, current_index: usize) -> usize {
    // Get one action that hasn't been tried yet
    let action = tree
        .get_node_mut(current_index)
        .unwrap()
        .remove_first_untried_action();

    // Clone current state and apply the action to get new state
    let current_state = tree.get_node(current_index).unwrap().state.clone();
    let new_state = action.apply_to(current_state.as_ref());

    // Add the new state as a child node in the tree
    let new_index = tree.add_child(current_index, new_state, action);
    return new_index;
}

/// Simulation/Rollout phase: play out the game with random actions.
///
/// From the given node's state, we randomly select actions until either:
/// 1. The game ends (terminal state reached)
/// 2. Maximum rollout depth is reached
///
/// The reward is discounted by gamma^depth to favor earlier rewards.
/// Using a random rollout policy is computationally efficient and provides
/// unbiased estimates of the value of a state (given sufficient samples).
///
/// # Arguments
/// * `tree` - Reference to the tree (for state access)
/// * `node_index` - Index of the node to simulate from
/// * `policy` - The rollout policy to use for action selection
///
/// # Returns
/// The discounted reward from the simulation
fn rollout(tree: &Tree, node_index: usize, policy: RolloutPolicy) -> Reward {
    // Clone the state so we don't modify the tree during simulation
    let mut current_state: Box<dyn State> = tree.get_node(node_index).unwrap().state.clone();
    let mut depth: i32 = 0;
    // Gamma < 1 ensures that rewards further in the future are worth less
    // This prevents infinite loops and focuses on near-term rewards
    let gamma: f64 = 0.95;

    // Keep simulating until game ends or we hit depth limit
    while depth < MAX_ROLLOUT_DEPTH && !current_state.is_game_ended() {
        let action_option = policy(current_state.as_mut());
        if let Some(action) = action_option {
            current_state = action.apply_to(current_state.as_ref());
            depth += 1;
        } else {
            // No legal actions available - shouldn't happen in normal play
            break;
        }
    }

    // Evaluate the final state and apply discount
    let reward = current_state.evaluate();
    return reward * gamma.powi(depth);
}

/// Backpropagation phase: update statistics along the path to root.
///
/// After a simulation completes, we update the Q-value (total reward) and
/// visit count for every node on the path from the expanded node back to
/// the root. This information is used by UCB1 in future selections.
///
/// The update adds the reward to each node's cumulative q-value and
/// increments the visit count. This allows the algorithm to learn from
/// the simulation results.
///
/// # Arguments
/// * `tree` - Mutable reference to the tree
/// * `node_index` - Starting node for backpropagation (typically the expanded node)
/// * `reward` - The reward from the simulation to propagate
fn backpropagate(tree: &mut Tree, node_index: usize, reward: Reward) {
    let mut current_index = node_index;

    loop {
        {
            // Update this node's statistics
            let current = tree.get_node_mut(current_index).unwrap();
            current.q += reward; // Add reward to cumulative total
            current.n += 1.0; // Increment visit count
        }

        // Move to parent and continue until we reach the root
        match tree.get_node(current_index).unwrap().parent_index {
            Some(parent_index) => {
                current_index = parent_index;
            }
            None => break, // Reached root - done
        }
    }
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
            let chosen = uct_best_child(&tree, 0, 1.4);
            assert_eq!(chosen, 1); // First child (index 1 in tree)
        }

        /// Tests that uct_best_child selects child with highest UCB1 value when all visited.
        ///
        /// # Steps
        /// 1. Create tree with two children
        /// 2. Mark both children as visited with different q/n values
        /// 3. Call uct_best_child and verify it picks the one with higher value
        ///
        /// # Expected Results
        /// - Child with higher average reward (q/n) should be selected
        #[test]
        fn test_uct_best_child_visited_selects_higher_value() {
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

            // Manually update children to have different q/n values
            // Child 1: q=10, n=10 -> average = 1.0
            // Child 2: q=20, n=10 -> average = 2.0
            {
                let child1 = tree.get_node_mut(1).unwrap();
                child1.q = 10.0;
                child1.n = 10.0;
            }
            {
                let child2 = tree.get_node_mut(2).unwrap();
                child2.q = 20.0;
                child2.n = 10.0;
            }
            // Update root's visit count as well (needed for UCB1 calculation)
            {
                let root = tree.get_node_mut(0).unwrap();
                root.n = 20.0;
            }

            // With exploration_constant=0, should select purely by exploitation
            // Child 2 has higher average reward, so it should be chosen
            let chosen = uct_best_child(&tree, 0, 0.0);
            assert_eq!(chosen, 2);
        }

        /// Tests that uct_best_child returns first child when children list is empty.
        ///
        /// # Steps
        /// 1. Create a tree (root has no children by default)
        /// 2. Call uct_best_child on root
        ///
        /// # Expected Results
        /// - Returns 0 (the root index) as a fallback
        #[test]
        fn test_uct_best_child_empty_children() {
            let state = Box::new(GridWorldState::new());
            let tree = Tree::new(state);

            // Root has no children, should return 0 as fallback
            let chosen = uct_best_child(&tree, 0, 1.4);
            assert_eq!(chosen, 0);
        }
    }

    /// Tests for Tree operations
    mod tree_tests {
        use super::*;

        /// Tests that get_node returns correct node by index.
        ///
        /// # Steps
        /// 1. Create a tree with root node
        /// 2. Call get_node with valid and invalid indices
        /// 3. Verify correct nodes are returned or None for invalid index
        ///
        /// # Expected Results
        /// - Valid index returns Some(node), invalid index returns None
        #[test]
        fn test_get_node_returns_correct_node() {
            let state = Box::new(GridWorldState::new());
            let tree = Tree::new(state);

            // Valid index should return Some
            let root = tree.get_node(0);
            assert!(root.is_some());

            // Invalid index should return None
            let invalid = tree.get_node(999);
            assert!(invalid.is_none());
        }

        /// Tests that get_node_mut returns correct mutable node by index.
        ///
        /// # Steps
        /// 1. Create a tree
        /// 2. Get mutable reference to root and modify it
        /// 3. Verify modification was applied
        ///
        /// # Expected Results
        /// - Mutable access works and modifications persist
        #[test]
        fn test_get_node_mut_modifies_node() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            // Get mutable reference and modify
            {
                let root = tree.get_node_mut(0).unwrap();
                root.q = 5.0;
                root.n = 10.0;
            }

            // Verify modification persisted
            let root = tree.get_node(0).unwrap();
            assert_eq!(root.q, 5.0);
            assert_eq!(root.n, 10.0);
        }

        /// Tests that add_child correctly creates new child node.
        ///
        /// # Steps
        /// 1. Create a tree with root
        /// 2. Add a child node
        /// 3. Verify child is created with correct properties
        ///
        /// # Expected Results
        /// - Child has correct parent index, state, and action
        #[test]
        fn test_add_child_creates_correct_node() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            // Add a child
            let child_state = Box::new(GridWorldState::new());
            let child_index = tree.add_child(
                0,
                child_state,
                Box::new(crate::grid_world::GridWorldAction::Right),
            );

            // Verify child exists and has correct properties
            let child = tree.get_node(child_index).unwrap();
            assert_eq!(child.parent_index, Some(0));
            assert!(child.causing_action.is_some());
            assert_eq!(child.causing_action.as_ref().unwrap().get_name(), "Right");

            // Verify root's children list includes this child
            let root = tree.get_node(0).unwrap();
            assert!(root.children.contains(&child_index));
        }
    }

    /// Tests for Node operations
    mod node_operation_tests {
        use super::*;

        /// Tests that remove_first_untried_action removes and returns the first action.
        ///
        /// # Steps
        /// 1. Create a tree with root that has untried actions
        /// 2. Remove first untried action
        /// 3. Verify it's removed from the list and returned
        ///
        /// # Expected Results
        /// - First action is removed and returned, list size decreases
        #[test]
        fn test_remove_first_untried_action() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);
            let root = tree.get_node_mut(0).unwrap();

            let initial_count = root.untried_actions.len();
            assert!(initial_count > 0);

            let action = root.remove_first_untried_action();
            assert_eq!(root.untried_actions.len(), initial_count - 1);
            // At starting position (1, 0), valid actions are Up and Down
            assert!(action.get_name() == "Up" || action.get_name() == "Down");
        }

        /// Tests that expand adds a new child node to the tree.
        ///
        /// # Steps
        /// 1. Create a tree
        /// 2. Call expand on the root
        /// 3. Verify a new child is created
        ///
        /// # Expected Results
        /// - New child node exists with updated state
        #[test]
        fn test_expand_adds_child_node() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            let initial_children_count = tree.get_node(0).unwrap().children.len();

            // Expand from root
            let new_index = expand(&mut tree, 0);

            // Verify new child was added
            let root = tree.get_node(0).unwrap();
            assert_eq!(root.children.len(), initial_children_count + 1);

            // Verify new node exists
            let new_node = tree.get_node(new_index).unwrap();
            assert!(new_node.parent_index == Some(0));
        }

        /// Tests that rollout performs simulation from a given node.
        ///
        /// # Steps
        /// 1. Create a tree with root at origin
        /// 2. Call rollout from root
        /// 3. Verify a reward is returned
        ///
        /// # Expected Results
        /// - Rollout returns a reward value (0 for non-terminal, 1 or -1 for terminal)
        #[test]
        fn test_rollout_returns_reward() {
            let state = Box::new(GridWorldState::new());
            let tree = Tree::new(state);

            let reward = rollout(&tree, 0, crate::policy::default);

            // Reward should be a valid value: 0, 1, or -1 (discounted)
            assert!(reward >= -1.0 && reward <= 1.0);
        }

        /// Tests that backpropagate updates node statistics correctly.
        ///
        /// # Steps
        /// 1. Create a tree with multiple nodes
        /// 2. Call backpropagate with a reward
        /// 3. Verify q and n values are updated along the path
        ///
        /// # Expected Results
        /// - All nodes on path have updated q and n values
        #[test]
        fn test_backpropagate_updates_statistics() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            // Add a child first
            let child_state = Box::new(GridWorldState::new());
            let child_index = tree.add_child(
                0,
                child_state,
                Box::new(crate::grid_world::GridWorldAction::Right),
            );

            // Backpropagate from child
            backpropagate(&mut tree, child_index, 1.0);

            // Verify child was updated
            let child = tree.get_node(child_index).unwrap();
            assert_eq!(child.q, 1.0);
            assert_eq!(child.n, 1.0);

            // Verify root was updated
            let root = tree.get_node(0).unwrap();
            assert_eq!(root.q, 1.0);
            assert_eq!(root.n, 1.0);
        }

        /// Tests that multiple backpropagations accumulate correctly.
        ///
        /// # Steps
        /// 1. Create a tree with a child
        /// 2. Call backpropagate multiple times
        /// 3. Verify q accumulates and n counts correctly
        ///
        /// # Expected Results
        /// - q is sum of rewards, n is count of simulations
        #[test]
        fn test_backpropagate_accumulates_multiple() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            // Add a child
            let child_state = Box::new(GridWorldState::new());
            let child_index = tree.add_child(
                0,
                child_state,
                Box::new(crate::grid_world::GridWorldAction::Right),
            );

            // Backpropagate multiple times with different rewards
            backpropagate(&mut tree, child_index, 1.0);
            backpropagate(&mut tree, child_index, -1.0);
            backpropagate(&mut tree, child_index, 0.5);

            // Verify accumulated values
            let child = tree.get_node(child_index).unwrap();
            assert_eq!(child.q, 1.0 - 1.0 + 0.5); // Sum = 0.5
            assert_eq!(child.n, 3.0); // 3 simulations

            let root = tree.get_node(0).unwrap();
            assert_eq!(root.q, 0.5);
            assert_eq!(root.n, 3.0);
        }

        /// Tests that select_and_expand returns terminal node directly.
        ///
        /// # Steps
        /// 1. Create a tree with root at terminal state (goal)
        /// 2. Call select_and_expand
        /// 3. Verify it returns the terminal node
        ///
        /// # Expected Results
        /// - Terminal node is returned without expansion
        #[test]
        fn test_select_and_expand_returns_terminal() {
            // Create a state at goal position (terminal)
            let mut terminal_state = GridWorldState::new();
            terminal_state.update_current_position(GridWorldState::GOAL_CELL);
            let state = Box::new(terminal_state);
            let mut tree = Tree::new(state);

            let result_index = select_and_expand(&mut tree, 0);

            // Should return the terminal node (root)
            assert_eq!(result_index, 0);
            assert!(tree.get_node(0).unwrap().is_terminal());
        }

        /// Tests that select_and_expand expands non-terminal nodes.
        ///
        /// # Steps
        /// 1. Create a tree with root at non-terminal state
        /// 2. Call select_and_expand
        /// 3. Verify a new child is created
        ///
        /// # Expected Results
        /// - A new child node is added to the tree
        #[test]
        fn test_select_and_expand_expands_nonterminal() {
            let state = Box::new(GridWorldState::new());
            let mut tree = Tree::new(state);

            let initial_children = tree.get_node(0).unwrap().children.len();

            select_and_expand(&mut tree, 0);

            let new_children = tree.get_node(0).unwrap().children.len();
            assert_eq!(new_children, initial_children + 1);
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

            // Verify the action name is valid (Up or Down from starting position)
            let name = action.get_name();
            assert!(
                name == "Up" || name == "Down",
                "Expected Up or Down, got {}",
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

            // Both should return valid action names (Up or Down from starting position)
            assert!(
                few_name == "Up" || few_name == "Down",
                "Expected Up or Down, got {}",
                few_name
            );
            assert!(
                many_name == "Up" || many_name == "Down",
                "Expected Up or Down, got {}",
                many_name
            );
        }
    }
}
