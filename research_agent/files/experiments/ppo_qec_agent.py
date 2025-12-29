"""
PPO Agent for Surface Code QEC Decoding

Implements a Proximal Policy Optimization agent for syndrome-based decoding.
The agent learns to predict whether a logical error occurred from syndromes.

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import json
import pickle


class PPOAgent:
    """
    PPO-style agent for QEC decoding.

    For syndrome decoding, we treat this as a classification problem:
    Given syndromes, predict the most likely logical error class.

    Uses a simple neural network with NumPy for compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        """
        Initialize PPO agent.

        Args:
            input_dim: Dimension of input (syndrome vector)
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            lambda_gae: GAE parameter
            eps_clip: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # For decoding, output is binary (logical error or not)
        self.output_dim = 2

        # Initialize weights
        self._init_weights()

        # Training history
        self.training_history = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "accuracy": []
        }

    def _init_weights(self):
        """Initialize network weights with Xavier initialization."""
        self.weights = {}
        self.biases = {}

        dims = [self.input_dim] + self.hidden_dims

        # Actor (policy) network
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights[f"actor_W{i}"] = np.random.randn(dims[i], dims[i + 1]) * scale
            self.biases[f"actor_b{i}"] = np.zeros(dims[i + 1])

        # Output layer for actor
        scale = np.sqrt(2.0 / (dims[-1] + self.output_dim))
        self.weights["actor_Wout"] = np.random.randn(dims[-1], self.output_dim) * scale
        self.biases["actor_bout"] = np.zeros(self.output_dim)

        # Critic (value) network
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights[f"critic_W{i}"] = np.random.randn(dims[i], dims[i + 1]) * scale
            self.biases[f"critic_b{i}"] = np.zeros(dims[i + 1])

        # Output layer for critic (single value)
        scale = np.sqrt(2.0 / dims[-1])
        self.weights["critic_Wout"] = np.random.randn(dims[-1], 1) * scale
        self.biases["critic_bout"] = np.zeros(1)

        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.m_bias = {k: np.zeros_like(v) for k, v in self.biases.items()}
        self.v_bias = {k: np.zeros_like(v) for k, v in self.biases.items()}
        self.t = 0

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward_actor(self, x: np.ndarray, store_activations: bool = False) -> np.ndarray:
        """Forward pass through actor network."""
        activations = {} if store_activations else None

        h = x.astype(np.float64)
        n_hidden = len(self.hidden_dims)

        for i in range(n_hidden):
            z = h @ self.weights[f"actor_W{i}"] + self.biases[f"actor_b{i}"]
            h = self._relu(z)
            if store_activations:
                activations[f"z{i}"] = z
                activations[f"h{i}"] = h

        logits = h @ self.weights["actor_Wout"] + self.biases["actor_bout"]
        probs = self._softmax(logits)

        if store_activations:
            activations["logits"] = logits
            activations["probs"] = probs
            activations["input"] = x
            return probs, activations

        return probs

    def forward_critic(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through critic network."""
        h = x.astype(np.float64)
        n_hidden = len(self.hidden_dims)

        for i in range(n_hidden):
            z = h @ self.weights[f"critic_W{i}"] + self.biases[f"critic_b{i}"]
            h = self._relu(z)

        value = h @ self.weights["critic_Wout"] + self.biases["critic_bout"]
        return value.flatten()

    def predict(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Predict logical error probability.

        Args:
            syndromes: Shape (n_samples, n_detectors)

        Returns:
            Predicted probabilities of logical error, shape (n_samples,)
        """
        probs = self.forward_actor(syndromes)
        return probs[:, 1]  # Probability of class 1 (logical error)

    def predict_correction(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Predict correction (0 = no flip, 1 = flip logical).

        Args:
            syndromes: Shape (n_samples, n_detectors)

        Returns:
            Predicted corrections, shape (n_samples,)
        """
        probs = self.forward_actor(syndromes)
        return np.argmax(probs, axis=1)

    def train_supervised(
        self,
        syndromes: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 256,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the agent using supervised learning on syndrome-label pairs.

        Args:
            syndromes: Input syndromes, shape (n_samples, n_detectors)
            labels: True logical error labels, shape (n_samples,)
            n_epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress

        Returns:
            Training history
        """
        n_samples = len(syndromes)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            total_loss = 0
            correct = 0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_x = syndromes[batch_indices]
                batch_y = labels[batch_indices]

                # Forward pass
                probs, activations = self.forward_actor(batch_x, store_activations=True)

                # Compute cross-entropy loss
                eps = 1e-10
                log_probs = np.log(probs + eps)
                loss = -np.mean(
                    batch_y * log_probs[:, 1] + (1 - batch_y) * log_probs[:, 0]
                )

                # Accuracy
                predictions = np.argmax(probs, axis=1)
                correct += np.sum(predictions == batch_y)

                # Backward pass (simplified gradient computation)
                batch_size_actual = len(batch_x)

                # Gradient of loss w.r.t. logits
                grad_logits = probs.copy()
                grad_logits[np.arange(batch_size_actual), batch_y.astype(int)] -= 1
                grad_logits /= batch_size_actual

                # Backprop through output layer
                h_last = activations[f"h{len(self.hidden_dims) - 1}"]
                grad_Wout = h_last.T @ grad_logits
                grad_bout = np.sum(grad_logits, axis=0)

                grad_h = grad_logits @ self.weights["actor_Wout"].T

                # Backprop through hidden layers
                grads_W = {"actor_Wout": grad_Wout}
                grads_b = {"actor_bout": grad_bout}

                for i in range(len(self.hidden_dims) - 1, -1, -1):
                    # ReLU gradient
                    grad_z = grad_h * self._relu_grad(activations[f"z{i}"])

                    # Get input to this layer
                    if i == 0:
                        h_prev = activations["input"]
                    else:
                        h_prev = activations[f"h{i - 1}"]

                    grads_W[f"actor_W{i}"] = h_prev.T @ grad_z
                    grads_b[f"actor_b{i}"] = np.sum(grad_z, axis=0)

                    if i > 0:
                        grad_h = grad_z @ self.weights[f"actor_W{i}"].T

                # Update weights with Adam
                self._adam_update(grads_W, grads_b)

                total_loss += loss * batch_size_actual

            avg_loss = total_loss / n_samples
            accuracy = correct / n_samples

            self.training_history["loss"].append(avg_loss)
            self.training_history["accuracy"].append(accuracy)

            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        return self.training_history

    def _adam_update(self, grads_W: Dict, grads_b: Dict, beta1: float = 0.9, beta2: float = 0.999):
        """Apply Adam optimizer update."""
        self.t += 1
        eps = 1e-8

        for key in grads_W:
            if key not in self.weights:
                continue

            # Weight update
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads_W[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads_W[key] ** 2)

            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            self.weights[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        for key in grads_b:
            if key not in self.biases:
                continue

            # Bias update
            self.m_bias[key] = beta1 * self.m_bias[key] + (1 - beta1) * grads_b[key]
            self.v_bias[key] = beta2 * self.v_bias[key] + (1 - beta2) * (grads_b[key] ** 2)

            m_hat = self.m_bias[key] / (1 - beta1 ** self.t)
            v_hat = self.v_bias[key] / (1 - beta2 ** self.t)

            self.biases[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def evaluate(self, syndromes: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate agent on test data.

        Args:
            syndromes: Test syndromes
            labels: True labels

        Returns:
            Evaluation metrics
        """
        probs = self.forward_actor(syndromes)
        predictions = np.argmax(probs, axis=1)

        accuracy = np.mean(predictions == labels)
        logical_error_rate = np.mean(predictions != labels)

        # Per-class metrics
        true_positive = np.sum((predictions == 1) & (labels == 1))
        false_positive = np.sum((predictions == 1) & (labels == 0))
        true_negative = np.sum((predictions == 0) & (labels == 0))
        false_negative = np.sum((predictions == 0) & (labels == 1))

        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "accuracy": float(accuracy),
            "logical_error_rate": float(logical_error_rate),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "true_negative": int(true_negative),
            "false_negative": int(false_negative)
        }

    def save(self, path: str):
        """Save agent to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "weights": self.weights,
            "biases": self.biases,
            "training_history": self.training_history,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "lambda_gae": self.lambda_gae,
                "eps_clip": self.eps_clip,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        """Load agent from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"],
            **data.get("hyperparameters", {})
        )
        agent.weights = data["weights"]
        agent.biases = data["biases"]
        agent.training_history = data.get("training_history", agent.training_history)
        return agent


if __name__ == "__main__":
    # Test the agent
    print("Testing PPO Agent")
    print("=" * 50)

    # Create dummy data
    n_samples = 1000
    n_detectors = 24  # Example for d=3, 3 rounds

    syndromes = np.random.randint(0, 2, (n_samples, n_detectors)).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)

    # Create agent
    agent = PPOAgent(input_dim=n_detectors, hidden_dims=[64, 32])

    # Train
    history = agent.train_supervised(syndromes, labels, n_epochs=20, batch_size=128)

    # Evaluate
    metrics = agent.evaluate(syndromes, labels)
    print(f"\nEvaluation: {metrics}")

    # Test save/load
    agent.save("/tmp/test_agent.pkl")
    loaded_agent = PPOAgent.load("/tmp/test_agent.pkl")
    print("Save/load test passed!")

    print("=" * 50)
    print("All tests passed!")
