# Reconnaissance d'images
_Reconnaissance d'images en utilisant keras et tensorflow_


# Fonctionnement du programme 
Pour faire tourner le programme, il suffit de récupérer un dataset sur kaggle, de le télécharger, et de changer les variables nécéssaires dans le code!

### Definition
Une IA est un système capable de répliquer le vivant, le biologique.
### Les différents types d'apprentissage
une I.A.a une méthode d'apprentissage appelée maching learning, celle ci est divisée en 3 parties:

-le supervisé, ce qu'on va faire, utilisation de "labels" ==> on dit si l'i.a. a raison et elle se corrige alors

-le non supervisé ==> on donne des données et l'i.a. apprend

-par renforcement ==> on établit un système de recompenses

La base de toute I.A. est le réseau de neurones. On part de plusieurs neurones, chacun donne un poids "weight" plus au moins important (compris entre 0 et 1) à des éléments de l'image analysée. La première "couche" de neurones est dite "input", puis on retrouve les "hidden", au milieu, et enfin les "output", dernière couche où le nombre de neurones correspond au nombre de possibilités (pour chat et chien, deux neurones)

### fonction:
l'i.a. va se baser sur plusieurs fonctions, une fonction etant un mécanisme qui prend une entrée et qui sort qqchose en sortie.
On étudiera par exemple la fonction "relu", pour les activations de couches. Relu est une fonction non linéaire, c'est à dire que si on relie toutes les images des antécédents, on obtient pas une droite.

## ce qu'on va faire

On va s'interresser à un MLP, alias Multilayer Perceptron (nom qui claque) qui est un "Approximateur Universel de Fonction" qui "rend presque exacte toutes les fonctions".

____________________________________________________________________________


