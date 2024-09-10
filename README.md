
# Controllable deep learning (controllabledl)

Ce répétoire contient une implementation de l'algorithme DeepCTRL présenté dans le papier [Controlling Neural Networks with Rule
Representations](https://arxiv.org/pdf/2106.07804)

Le but de ce projet était d'appliquer une méthode de pricing en combinant deux approches complémentaires. D'une part, le pricing traditionnel suit des règles de simplicité, de conformité et d'explicabilité, mais manque parfois de précision. D'autre part, le pricing basé sur le machine learning offre une grande précision et personnalisation, mais présente des défis en termes de conformité et de transparence. La solution proposée est donc un modèle de pricing "hybride", qui combine les avantages des deux approches, permettant à la fois une personnalisation accrue et une meilleure conformité aux règles.

![Deep Control](https://github.com/AbdellahRafik/DeepCTRL/blob/main/DeepCTRL/ressources/Deepctrl.png)

DEEPCTRL est une méthode permettant l'incorporation contrôlée d'une règle au sein du processus d'apprentissage. Elle introduit deux passages distincts pour la relation entrée-sortie : un encodeur de données et un encodeur de règles, qui produisent respectivement deux représentations latentes, zd et zr. Ces deux représentations sont ensuite concaténées de manière stochastique avec un paramètre de contrôle, α, pour former une représentation unique, z. Cette représentation z est ensuite transmise à un bloc décisionnel, où des objectifs distincts sont définis pour chaque représentation, Lrule pour les règles et Ltask pour la tâche à accomplir, ces objectifs étant pondérés par α.



## Dependencies
- python 3.7
- pytorch 1.6.0


## Documentation

Papier de recherche : [Controlling Neural Networks with Rule
Representations
](https://arxiv.org/pdf/2106.07804)


## Data