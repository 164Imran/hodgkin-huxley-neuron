# Hodgkin-Huxley Neuron Simulation

Before building a neural network, I wanted to understand what a neuron actually *is* — not as an abstraction, but as a physical object: ion channels opening and closing, a membrane charging and discharging, a spike that either fires or doesn't.

The Hodgkin-Huxley model is the closest thing we have to a ground truth for that. Four coupled differential equations, integrated here with RK4, that reproduce the exact shape of an action potential.

---

## The bigger picture

This is the first step toward something larger: a **network of Hodgkin-Huxley neurons**.

To get there, the next problems to solve are:

- **Synaptic connections** — modelling how one neuron's spike triggers a current in the next (excitatory / inhibitory synapses, transmission delay)
- **Plasticity** — making the connection weights adapt over time, not just simulate fixed anatomy

On plasticity specifically, I'm curious about whether **ant colony optimization** could be relevant here. ACO is a collective memory mechanism: pheromone trails reinforce frequently-used paths and fade otherwise — which structurally mirrors Hebbian learning (*neurons that fire together, wire together*). Whether it maps cleanly onto spike-timing dynamics is an open question, but worth exploring.

---

## Stack

```
numpy · matplotlib
```

```bash
python NN/hodgkin_huxley_neuro.py
```
