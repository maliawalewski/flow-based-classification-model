# Flow-Based Classification with RealNVP

This project compares classification, generative modeling, and hybrid modeling on the two moons dataset using neural networks and flow-based models (RealNVP).
It was done as a homework for my Deep Generative Modeling class. 

## What's Inside

- Train a fully connected neural classifier on 2D moons
- Evaluate generalization with decision boundary and accuracy plots
- Train a flow-based generative model (RealNVP) using maximum likelihood
- Explore learned densities and out-of-distribution confidence
- Build a hybrid model that shares an encoder across both tasks
- Visualize training dynamics and decision behavior

## Tools
- PyTorch (for custom models)
- NumPy, Matplotlib
- `make_moons` dataset from scikit-learn

## Key Results
- Classifier achieved ~100% accuracy
- Flow model learned smooth densities
- Hybrid model explored shared representations, but showed instability in convergence — future work could explore better loss balancing and model interpretability

## Structure
- `SimpleNN`: Feedforward classifier
- `RealNVP`: Flow model using affine coupling
- `HybridFlowClassifier`: Combines both under shared encoder

> This project was completed as part of an advanced ML assignment in a university course.
