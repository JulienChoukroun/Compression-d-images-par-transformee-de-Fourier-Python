Ce projet est réalisé dans le cadre de la formation de cycle d'ingénieur (première année) de l'Ecole Polytechnique de l'Université Côte d'Azur.
***
# Compression d'images par transformée de Fourier

## Présentation
Ce projet a été réalisé avec le langage Python.

La transmission des informations est d’autant plus facile que le nombre d’informations est faible.
Il est donc avantageux de réduire la taille d’un fichier pour l’envoyer plus rapidement et plus facilement. Une technique employée dans de nombreux domaines est la compression des informations qui permet en effet, une réduction importante de la quantité de données.
Voici l’algorithme permettant la compression d’une image :
![alt text](https://github.com/JulienChoukroun/PythonMAM3S5/blob/master/Documentation/algo.png "Algorithme")

Une fois après avoir implémenté et codé la compression, il n’est pas difficile d’aborder la décompression. Il suffit de faire le chemin inverse de la compression en appliquant la DCT inverse. 
