# treeq

<p align="center">
<img src="./tree.jpg" alt="tree" width=300/>
</p>

`treeq` is a small educational experiment for learning how neural networks can learn structured relationships.

Current Implementation follows a key paper:

- Back-propagating Errors [Rumelhart, Hinton & Williams .1986](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)


It uses two small, isomorphic family trees — one English and one Italian — and trains a model to answer questions such as:

```text
John → father → David
David → sister → Sarah
James → brother → Robert
```

The family trees are defined using only three primitives:

- `parents`
- `spouse`
- `sex`

All other relationships are derived from these primitives. The generated relationships are then converted into training triples and used to train a small neural network.

The main experiment compares two models:

- a **24-dimensional person representation** as the control
- a **6-dimensional person representation** as the bottleneck

The goal is to see whether the network can compress the identities and structure of the family trees into a much smaller representation while still learning the relationships between people.

This is intentionally a small, hackable project for educational purposes.


>  This project really gives a proper understanding of relationships 😋
