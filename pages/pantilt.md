---
title: "Apprendre la géométrie d'un Pan-Tilt"
output:
    md_extensions: +raw_html
---

<img src="imgs/pantilt.png" width="200" class="float-end" />

Dans ce TP, nous allons apprendre le **modèle géométrique** inverse d'un *pan-tilt*.

## 0. Prise en main

<div class="step">

Tout d'abord, téléchargez l'archive ci-dessous et décompressez-la:

<div class="alert alert-info d-flex align-items-center justify-content-center">
<img src="imgs/archive.png" />
<div>
<big><a href="files/pan_tilt.zip">Téléchargez l'archive pan_tilt.zip</a></big>
<br/>
<em>Lisez les instructions ci-dessous</em>
</div>
</div>

</div>
<div class="step">

Installez les dépendances:

```bash
pip install numpy pygame pybullet onshape-to-robot transforms3d scipy
```

</div>
<div class="step">

Lancez:

```bash
python sim.py
```

Et observez le résultat.

</div>

## 1. Modèle géométrique direct

Le modèle du pan-tilt est le suivant:

<center>
<img src="imgs/pan_tilt.png" width="150" />
</center>

<div class="step">

Dans `model.py`, implémentez la méthode `direct`, qui prend en entrée les deux angles du robot, et produit la matrice de transformation ${}^w T_e$ 4x4 allant de l’effecteur au monde.

Pour tester, lancez le programme de cette façon:

```bash
python sim.py -m direct
```

</div>

## 2. Intersection avec le sol

<div class="step">

Implémentez maintenant la méthode `laser`, qui calcule l’intersection au sol ($z=0$) d’un laser qui partirait de l’axe x de l’effecteur.

Pour tester, lancez le programme de cette façon:

```bash
python sim.py -m laser
```

Voici le résultat que vous devriez obtenir:

<center>
<img src="imgs/pan_tilt_laser.png" width="400" />
</center>

</div>

## 3. Apprentissage de la géométrie inverse

Dans cette partie, l'objectif est d'apprendre au laser à viser un point cible. Pour cela, nous allons utiliser la fonction `laser` créée précédemment, et essayer d'apprendre l'inverse de cette fonction à l'aide d'un réseau de neurones.

### 3.1 Perceptron multi-couches

<div class="step">

Exécutez `learn_example.py`, lisez son code ainsi que celui de `mlp.py` et répondez aux quesitons suivantes.

<div class="step">
Combien de couches cachées y'a-t-il dans ce réseau ?
</div>

<div class="step">
Combien de paramètres entraînables y'a-t-il dans ce réseau ?
</div>

<div class="step">

Dans la ligne `net = MLP(1, 1)`, que signifient les deux arguments `1` et `1` ?

</div>

<div class="step">

Quelle fonction de perte est utilisée ?

</div>

## 3.2 Entraînement

En vous inspirant de `learn_example`, créez un fichier `learn.py` dans lequel:

<div class="step">

Créez un perceptron multi-couche, prenant 2 entrées $(x, y)$ et produisant 2 sorties $(\alpha, \beta)$.

</div>

Pour chaque époque (1000 au total):

<div class="step">

* Génère `batch_size` angles $(\alpha, \beta)$, et calcule la position du laser à l'aide de la fonction `laser`.

</div>
<div class="step">

* Calculez la perte entre la position prédite et la position cible, puis effectuez une rétropropagation pour mettre à jour les poids du réseau.

</div>
<div class="step">

Enfin, sauvez les poids du réseau à la fin de l'entraînement.

</div>

## 3.3 Inférence

<div class="step">

Dans `model.py`, implémentez la méthode `inverse_nn`. Elle prend en argument une cible et retourne les angles qui doivent permettre au *pan-tilt* de la regarder. Pour cela, il suffit de charger les poids du réseau que vous avez entraîné précédemment, et d'effectuer une inférence.

Vous testerez votre code de la même façon que précédemment:

```bash
python sim.py -m inverse_nn
```

</div>

## 3.4 Courbe d'apprentissage

<div class="step">

Modifiez le code d'entraînement pour afficher la courbe d'apprentissage, c'est à dire la perte en fonction du nombre d'époques.

</div>

## 3.5 Une amélioration

Remarquez ce qu'il se passe lorsque votre pan-tilt regarde *derrière* lui, c'est à dire en $x<0$ et $y=0$. Le problème ici vient de la difficulté pour le réseau d'apprendre à cause de la discontinuité des angles.

<div class="step">

Lors de l'apprentissage, remplacez les angles $\alpha$ et $\beta$ par leurs sinus et cosinus. Par exemple, au lieu d'apprendre à prédire $\alpha$, vous apprendrez à prédire $\sin(\alpha)$ et $\cos(\alpha)$. De cette façon, le réseau n'aura plus de discontinuité à apprendre.

</div>
<div class="step">

Lors de l'inférence, vous devrez faire le calcul inverse pour retrouver les angles à partir de leurs sinus et cosinus. Par exemple, $\alpha = \arctan2(\sin(\alpha), \cos(\alpha))$.

</div>

## 4 Modèle inverse analytique

<div class="step">

Implémentez la méthode `inverse` dans `model.py`, qui calcule l'inverse de la fonction `laser` de manière analytique, c'est à dire sans apprentissage.

Vous testerez votre code de la même façon que précédemment:

```bash
python sim.py -m inverse
```

</div>