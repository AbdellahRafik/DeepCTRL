
# Controllable deep learning (controllabledl)

This repository contains an implementation of the DeepCTRL algorithm presented in the paper [Controlling Neural Networks with Rule
Representations](https://arxiv.org/pdf/2106.07804)

The goal of this project was to apply a pricing method by combining two complementary approaches. On the one hand, traditional pricing follows rules of simplicity, compliance, and explainability but sometimes lacks precision. On the other hand, machine learning-based pricing offers high precision and customization but faces challenges in terms of compliance and transparency. The proposed solution is a "hybrid" pricing model that combines the benefits of both approaches, allowing for increased personalization and better compliance with rules.

![Deep Control](https://github.com/AbdellahRafik/DeepCTRL/blob/main/DeepCTRL/ressources/Deepctrl.png)

DEEPCTRL is a method that enables controlled incorporation of a rule within the learning process. It introduces two distinct pathways for the input-output relationship: a data encoder and a rule encoder, which respectively produce two latent representations, zd and zr. These two representations are then stochastically concatenated with a control parameter, α, to form a unique representation, z. This z representation is then passed to a decision block, where distinct objectives are defined for each representation: Lrule for the rules and Ltask for the task at hand, these objectives being weighted by α.



## Dependencies
- python 3.7
- pytorch 1.6.0


## Documentation

Papier de recherche : [Controlling Neural Networks with Rule
Representations
](https://arxiv.org/pdf/2106.07804)


## Data