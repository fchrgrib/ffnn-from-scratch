import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class FFNN:
    """
    Feedforward Neural Network implementation from scratch.
    """
    
    def __init__(self, layer_sizes, activation_functions, loss_function, weight_init_method='random_uniform', 
                 weight_init_params=None):
        """
        Initialize the FFNN model.
        
        Parameters:
        -----------
        layer_sizes : list
            Number of neurons in each layer (including input and output layers)
        activation_functions : list
            Activation functions for each layer (except input layer)
        loss_function : str
            Loss function to use for training
        weight_init_method : str
            Method for weight initialization
        weight_init_params : dict
            Parameters for weight initialization
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Validation
        if len(activation_functions) != self.num_layers - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")
        
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        
        # Default weight initialization parameters
        if weight_init_params is None:
            if weight_init_method == 'random_uniform':
                weight_init_params = {'lower_bound': -0.5, 'upper_bound': 0.5, 'seed': 42}
            elif weight_init_method == 'random_normal':
                weight_init_params = {'mean': 0, 'variance': 0.1, 'seed': 42}
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []
        self._initialize_weights(weight_init_method, weight_init_params)
        
        # For storing intermediate values during forward/backward pass
        self.z_values = []  # Pre-activation values
        self.a_values = []  # Post-activation values
    
    def _initialize_weights(self, method, params):
        """
        Initialize weights and biases based on the specified method.
        
        Parameters:
        -----------
        method : str
            Method for weight initialization ('zero', 'random_uniform', 'random_normal')
        params : dict
            Parameters for weight initialization
        """
        match method:
            case 'zero':
                for i in range(1, self.num_layers):
                    self.weights.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])))
                    self.biases.append(np.zeros((self.layer_sizes[i], 1)))
                    self.weight_gradients.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])))
                    self.bias_gradients.append(np.zeros((self.layer_sizes[i], 1)))
            case 'random_uniform':
                lower_bound = params.get('lower_bound', -0.5)
                upper_bound = params.get('upper_bound', 0.5)
                seed = params.get('seed', None)
                
                if seed is not None:
                    np.random.seed(seed)
                
                for i in range(1, self.num_layers):
                    self.weights.append(np.random.uniform(
                        lower_bound, upper_bound, size=(self.layer_sizes[i], self.layer_sizes[i-1])))
                    self.biases.append(np.random.uniform(
                        lower_bound, upper_bound, size=(self.layer_sizes[i], 1)))
                    self.weight_gradients.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])))
                    self.bias_gradients.append(np.zeros((self.layer_sizes[i], 1)))
            case 'random_normal':
                mean = params.get('mean', 0)
                variance = params.get('variance', 0.1)
                std_dev = np.sqrt(variance)
                seed = params.get('seed', None)
                
                if seed is not None:
                    np.random.seed(seed)
                
                for i in range(1, self.num_layers):
                    self.weights.append(np.random.normal(
                        mean, std_dev, size=(self.layer_sizes[i], self.layer_sizes[i-1])))
                    self.biases.append(np.random.normal(
                        mean, std_dev, size=(self.layer_sizes[i], 1)))
                    self.weight_gradients.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])))
                    self.bias_gradients.append(np.zeros((self.layer_sizes[i], 1)))
            case _:
                raise ValueError(f"Unsupported weight initialization method: {method}")
    
    def _activate(self, z, activation_function):
        """
        Apply activation function.
        
        Parameters:
        -----------
        z : numpy.ndarray
            Pre-activation values
        activation_function : str
            Name of the activation function
            
        Returns:
        --------
        numpy.ndarray
            Activated values
        """
        match activation_function:
            case 'linear':
                return z
            case 'relu':
                return np.maximum(0, z)
            case 'sigmoid':
                z_safe = np.clip(z, -500, 500)  # Prevent overflow
                return 1 / (1 + np.exp(-z_safe))
            case 'tanh':
                return np.tanh(z)
            case 'softmax':
                shifted_z = z - np.max(z, axis=0, keepdims=True) # Prevent overflow
                exp_z = np.exp(shifted_z)
                return exp_z / np.sum(exp_z, axis=0, keepdims=True)
            case _:
                raise ValueError(f"Unsupported activation function: {activation_function}")

    def _activate_derivative(self, z, activation_function):
        """
        Compute the derivative of the activation function.

        Parameters:
        -----------
        z : numpy.ndarray
            Pre-activation values
        activation_function : str
            Name of the activation function

        Returns:
        --------
        numpy.ndarray
            Derivatives of the activation function
        """
        match activation_function:
            case 'linear':
                return np.ones_like(z)
            case 'relu':
                return (z > 0).astype(float)
            case 'sigmoid':
                activated_values = self._activate(z, 'sigmoid')
                return activated_values * (1 - activated_values)
            case 'tanh':
                return 1 - np.tanh(z) ** 2
            case 'softmax':
                activated_values = self._activate(z, 'softmax')
                num_classes, batch_size = activated_values.shape
                jacobians = []

                for i in range(batch_size):
                    s = activated_values[:, i].reshape(-1, 1)
                    
                    jacobian = np.zeros((num_classes, num_classes))
                    for i in range(num_classes):
                        for j in range(num_classes):
                            if i == j:
                                jacobian[i, j] = s[i, 0] * (1 - s[i, 0])
                            else:
                                jacobian[i, j] = -s[i, 0] * s[j, 0]
                    jacobians.append(jacobian)
                return jacobians
            case _:
                raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute the loss between true and predicted values.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        float
            Loss value
        """
        n_samples = y_true.shape[1]
        
        match self.loss_function:
            case 'mse':
                return np.mean(np.sum((y_true - y_pred)**2, axis=0)) / 2
            case 'binary_crossentropy':
                # Clip to avoid log(0)
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n_samples
            case 'categorical_crossentropy':
                # Clip to avoid log(0)
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -np.sum(y_true * np.log(y_pred)) / n_samples
            case _:
                raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def _compute_loss_derivative(self, y_true, y_pred):
        """
        Compute the derivative of the loss function.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        numpy.ndarray
            Derivative of the loss function
        """
        n_samples = y_true.shape[1]
        
        match self.loss_function:
            case 'mse':
                return (y_pred - y_true) / n_samples
            case 'binary_crossentropy':
                # Clip to avoid division by 0
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n_samples
            case 'categorical_crossentropy':
                # For softmax activation function
                if self.activation_functions[-1] == 'softmax':
                    return (y_pred - y_true) / n_samples
                
                # For other activation functions
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -y_true / y_pred / n_samples
            case _:
                raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def forward(self, X):
        """
        Perform forward propagation.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (shape: features x samples)
            
        Returns:
        --------
        numpy.ndarray
            Predictions from the output layer
        """
        # Reset stored intermediate values
        self.z_values = []
        self.a_values = []
        
        # First activation is the input
        a = X
        self.a_values.append(a)
        
        # Forward propagation through each layer
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self._activate(z, self.activation_functions[i])
            
            self.z_values.append(z)
            self.a_values.append(a)
        
        return a  # Output of the last layer
    
    def backward(self, X, y):
        """
        Perform backward propagation to compute gradients.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (shape: features x samples)
        y : numpy.ndarray
            Target values (shape: outputs x samples)
        """
        n_samples = X.shape[1]
        
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute the initial error (delta) from the loss function
        delta = self._compute_loss_derivative(y, y_pred)
        
        # Handle the special case for softmax + categorical cross-entropy
        if self.activation_functions[-1] == 'softmax':
            if self.loss_function == 'categorical_crossentropy':
                pass
            else:
                jacobians = self._activate_derivative(
                    self.z_values[-1],
                    self.activation_functions[-1]
                )
                new_delta = np.zeros_like(delta)
                
                for k in range(n_samples):
                    new_delta[:, k] = np.dot(jacobians[k], delta[:, k].reshape(-1, 1)).flatten()
                
                delta = new_delta
        else:
            # For other combinations, multiply by the activation derivative
            delta = delta * self._activate_derivative(self.z_values[-1], 
                                                     self.activation_functions[-1])
        
        # Backpropagate through each layer
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients for this layer
            self.weight_gradients[i] = np.dot(delta, self.a_values[i].T) / n_samples
            self.bias_gradients[i] = np.sum(delta, axis=1, keepdims=True) / n_samples
            
            # Compute delta for the previous layer (if not the input layer)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta)
                delta = delta * self._activate_derivative(self.z_values[i-1], 
                                                         self.activation_functions[i-1])
    
    def update_weights(self, learning_rate):
        """
        Update weights and biases using gradient descent.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.weight_gradients[i]
            self.biases[i] -= learning_rate * self.bias_gradients[i]
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, learning_rate=0.01, 
              epochs=100, verbose=1):
        """
        Train the FFNN model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training input data (shape: features x samples)
        y_train : numpy.ndarray
            Training target values (shape: outputs x samples)
        X_val : numpy.ndarray, optional
            Validation input data
        y_val : numpy.ndarray, optional
            Validation target values
        batch_size : int
            Size of batches for mini-batch gradient descent
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level (0: no output, 1: progress bar)
            
        Returns:
        --------
        dict
            Training history (loss and validation loss)
        """
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                batch_loss = self._compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Backward pass
                self.backward(X_batch, y_batch)
                
                # Update weights
                self.update_weights(learning_rate)
            
            # Record training loss
            history['train_loss'].append(epoch_loss)
            
            # Compute validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self._compute_loss(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
            
            # Print progress
            if verbose == 1:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (shape: features x samples)
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        return self.forward(X)
    
    def visualize_model(self):
        """
        Visualize the model structure with weights and gradients as a graph.
        """
        G = nx.DiGraph()
        
        # Add nodes for each layer
        layer_nodes = []
        for l in range(self.num_layers):
            layer_nodes.append([])
            for n in range(self.layer_sizes[l]):
                node_id = f"L{l}N{n}"
                G.add_node(node_id, layer=l, neuron=n)
                layer_nodes[l].append(node_id)
        
        # Add edges with weights and gradients
        for l in range(self.num_layers - 1):
            for i in range(self.layer_sizes[l]):
                for j in range(self.layer_sizes[l+1]):
                    weight = self.weights[l][j, i]
                    gradient = self.weight_gradients[l][j, i]
                    G.add_edge(
                        layer_nodes[l][i],
                        layer_nodes[l+1][j],
                        weight=weight,
                        gradient=gradient
                    )
        
        # Create positions for visualization
        pos = {}
        for l in range(self.num_layers):
            for n in range(self.layer_sizes[l]):
                pos[f"L{l}N{n}"] = (l, n - self.layer_sizes[l]/2)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        colors = []
        for node in G.nodes():
            layer = int(node[1])
            if layer == 0:
                colors.append('blue')  # Input layer
            elif layer == self.num_layers - 1:
                colors.append('red')   # Output layer
            else:
                colors.append('green') # Hidden layers
        
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, alpha=0.8)
        
        # Draw edges with color based on weight value
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for coloring
        weight_abs = [abs(w) for w in weights]
        max_weight = max(weight_abs) if weight_abs else 1.0
        norm_weights = [abs(w)/max_weight for w in weights]
        
        cmap = plt.cm.Blues
        nx.draw_networkx_edges(G, pos, width=2, edge_color=norm_weights, edge_cmap=cmap)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Neural Network Structure with Weights and Gradients")
        plt.axis('off')
        plt.show()
    
    def plot_weight_distribution(self, layers=None):
        """
        Plot the distribution of weights for specified layers.
        
        Parameters:
        -----------
        layers : list, optional
            List of layer indices to plot. If None, all layers are plotted.
        """
        if layers is None:
            layers = list(range(len(self.weights)))
        
        n_layers = len(layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))
        
        # Handle case with only one layer
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.weights):
                print(f"Warning: Layer index {layer_idx} is out of range. Skipping.")
                continue
                
            weights = self.weights[layer_idx].flatten()
            axes[i].hist(weights, bins=30, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx+1} Weights")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distribution(self, layers=None):
        """
        Plot the distribution of weight gradients for specified layers.
        
        Parameters:
        -----------
        layers : list, optional
            List of layer indices to plot. If None, all layers are plotted.
        """
        if layers is None:
            layers = list(range(len(self.weight_gradients)))
        
        n_layers = len(layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))
        
        # Handle case with only one layer
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers):
            if layer_idx < 0 or layer_idx >= len(self.weight_gradients):
                print(f"Warning: Layer index {layer_idx} is out of range. Skipping.")
                continue
                
            gradients = self.weight_gradients[layer_idx].flatten()
            axes[i].hist(gradients, bins=30, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx+1} Gradients")
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency")
            
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activation_functions': self.activation_functions,
            'loss_function': self.loss_function,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        
        np.save(filepath, model_data, allow_pickle=True)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
            
        Returns:
        --------
        FFNN
            Loaded model
        """
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create a new model instance
        model = cls(
            layer_sizes=model_data['layer_sizes'],
            activation_functions=model_data['activation_functions'],
            loss_function=model_data['loss_function'],
            weight_init_method='zero'  # Will be overwritten
        )
        
        # Replace the weights and biases
        model.weights = [np.array(w) for w in model_data['weights']]
        model.biases = [np.array(b) for b in model_data['biases']]
        
        # Initialize gradients with zeros
        model.weight_gradients = [np.zeros_like(w) for w in model.weights]
        model.bias_gradients = [np.zeros_like(b) for b in model.biases]
        
        print(f"Model loaded from {filepath}")
        return model
