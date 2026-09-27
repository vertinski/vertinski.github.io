# ABS MLP Playground

An interactive, in-browser playground for training a tiny multilayer perceptron with **absolute value (ABS) activations**. Draw your own 2D dataset, pick a training setup, and watch the network fold the plane into a decision boundary in real time.

**Live demo:** `https://<your-username>.github.io/<your-repo>/`
**Blog post:** `https://<your-username>.github.io/<your-repo>/blog.html`

No build step, no frameworks, no dependencies: the network, backpropagation, optimizer and rendering are all hand-written JavaScript in a single HTML file.

## Why ABS?

A ReLU unit cuts the input plane along a line and discards one side. An ABS unit **folds** the plane along that line, reflecting one half onto the other: its output `|w·x + b|` is the scaled distance to the fold line.

This makes XOR trivial. The single unit `|x₁ − x₂|` is 0 for (0,0) and (1,1) and 1 for (0,1) and (1,0), so one hidden unit solves XOR outright. With more units and layers, the decision boundary becomes a polygon built from folds, and folds of folds. ABS also has no dead units: its gradient is ±1 everywhere except exactly at zero.

## Features

**Live visualization**

- Decision surface heatmap with the 0.5 decision boundary
- Every hidden unit’s fold drawn on the surface: straight lines for the first layer, bent curves for deeper layers
- Network diagram with weight magnitude and sign
- Log-scale loss curve, with markers for the automatic decay schedule
- Per-sample predictions table, including how often each sample is picked during training

**Dataset editing**

- Click to add a point of the selected class, Shift-click for the other class
- Click a point to remove it
- Restore the XOR points or clear the plane

**Model**

- 1 to 4 hidden layers, 1 to 10 units per layer, ABS activation on every hidden layer
- Sigmoid output for probabilistic losses, raw score for hinge losses

**Losses**

- Binary cross-entropy
- Mean squared error
- Mean absolute error
- Focal loss (γ = 2)
- Hinge
- Squared hinge

**Optimization**

- Full-batch gradient descent or mini-batch SGD (batch 1, 2, 4, 8, 16)
- Momentum: 0, 0.5, 0.9
- Weight decay, applied to each hidden unit’s weights and bias together so decay never moves a fold line
- Parameter noise: a Gaussian kick to every weight and bias after each step
- Automatic weight decay drop: decay switches from 0.001 to 0.0001 once the loss falls below an adjustable threshold

**Sampling strategies (mini-batch)**

|Mode                   |Behavior                                                              |
|-----------------------|----------------------------------------------------------------------|
|Shuffled               |Random order each epoch, every sample used once                       |
|Hardest first          |Each epoch, samples ordered by current loss, highest first            |
|Loss-weighted (repeats)|Drawn with replacement, probability proportional to current loss      |
|Hardest only           |Every step takes the highest-loss samples (online hard example mining)|

## Recommended settings

These defaults worked best on hand-drawn double spirals and similar hard shapes, and are what the page loads with:

|Setting        |Value                                  |
|---------------|---------------------------------------|
|Hidden layers  |1                                      |
|Units per layer|10                                     |
|Learning rate  |0.01                                   |
|Loss           |Mean squared error                     |
|Batch          |1 (SGD)                                |
|Sampling       |Loss-weighted (repeats)                |
|Momentum       |0                                      |
|Weight decay   |0.001, dropping to 0.0001 automatically|
|Parameter noise|0.002                                  |
|Steps per frame|100                                    |

### Choosing the decay drop threshold

The threshold depends on the dataset. Train with strong decay first, watch where the loss plateaus, and set the threshold below the plateau average, near the bottom of its noise band. Too early, and the network loses the smoothing phase before it has found the coarse shape. Too late, and steps are wasted on the plateau. In practice, the drop consistently makes the loss fall faster and settle far lower than training with a fixed decay.

## Implementation notes

- **Forward and backward passes** are written by hand for an arbitrary number of layers. Gradients were verified against numerical finite differences for 1 to 4 hidden layers across several losses.
- **Initialization:** first-layer weights are uniform in [−1, 1], hidden biases uniform in [−0.5, 0.5]. Deeper hidden layers use variance 1/fan-in, which preserves signal scale through ABS since |a|² = a². Output weights are uniform in ±1.5/√fan-in with a zero bias.
- **Weight decay** is coupled L2 (added to the gradient, so it passes through momentum). Hidden units decay (w, b) together; the output bias is not decayed.
- **Rendering** uses the Canvas 2D API. The decision boundary and deeper-layer folds are traced with marching squares.
- **Fonts** load from Google Fonts, with a system font fallback when offline.

## License

MIT license 
