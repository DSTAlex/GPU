Question 1
Combien de float par bloc peut-on instancier au maximum dans la shared memory ?
	49152 / 4 = 12288
	On peut donc metre 12288 float dans la memoire partagé.

Quelle est la taille maximale d’un bloc 3D dont les dimensions sont égales ?
	Nous avons 1024 thread par bloc. racine cubique de 1024 = 10.07.
	On peut donc construire un bloc de 10x10x10 et il y aura 24 thread non utilisés.  

Selon la version compute capability, le GPU a-t-il des Tensor Cores ?
	non car il faut une compute capability de 8.0


Question 2
En modifiant le nombre de blocs à 4 et le nombre de threads par bloc à 32, combien de fois le
message “Hello World” est-il affiché? Quels sont les ID maximum des threads et des blocs ? 
    32 * 8 = 128
    Il y aura 128 messages.

Y a-t-il un ordre particulier des affichages en fonction des blocs ou des threads ?
    Certain se message se suivent mais l'orde des blocs et des thread semble aleatoire.
    Ceci est du a la parallélisation.


Question 3
Combien de blocs sont-ils nécessaires pour traiter l’intégralité des tableaux de taille N?
    Il faut la borne superieur de (N / thread_per_bloc) dans notre cas il nous faut 32 blocs.

Combien de threads sont inactifs (appartiennent à un bloc actif mais n’effectue aucun calcul)
    0.75 * 32 = 24
    Il y a 24 thread inactifs.

Étant donné les charactéristiques du GPU, quelle est la taille maximale N des tableaux que l’on peut traiter en parallèle avec cet algorithme Map ?
    1024 * 1024 = 1048576 car l'algorithme ne prend pas en compte les 3dimentions des blocs
    On peut faire un tableau de 1 048 576 en parallèle.


Question 4
Quelle est la valeur de la stride ?
    8 * 32 = 256

Combien y a-t-il de thread au total ? 
    8 * 32 = 256

Environ combien d’addition effectue un thread ?
    Chaque thread fait 4 addition sauf les 24 dernier du dernier bloc qui en font 3.
