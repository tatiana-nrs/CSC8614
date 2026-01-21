# **rapport.md — TP3 csc8614-language-models**
NIAURONIS Tatiana – FIPA 3A  
CSC8614 – TP3

---

### **Question 1**

```bash
conda create -n csc8614_tp3 python=3.10
conda activate csc8614_tp3
pip install -r requirements.txt
```

```
OS: macOS 14.3.1 (arm64)
Python : 3.10.19
torch==2.9.1
tiktoken==0.12.0
tqdm==4.67.1
pandas==2.3.3
matplotlib==3.10.8
tensorflow==2.20.0
jupyterlab==4.5.1
```
---

# Exercice 4 

### **Question 1**

Oui il y'a une différence. Dans le modèle original, les couches sont des Linear(...). Après injection LoRA, ces mêmes couches apparaissent comme LinearWithLoRA(...).  Donc oui, on voit bien que cela a fonctionné.

### **Question 2**

On a Trainable params : 1,327,104, All params : 164,364,288 et Trainable % : 0.81%.

En effet, on a gelé les poids du modéle original avec requires_grad=False donc seules les matrices LoRA A et B sont entrainées ici (rank k).

### **Question 3**

Après l’ajout de la tête de classification, on observe une différence par rapport à la situation d'avant. Le nombre de paramètres entraînables augmente (de 0,81 % à environ 1,06 %) car en plus des paramètres LoRA (matrices A et B), la nouvelle tête de classification est entraînable. Cette augmentation reste faible car la tête de classification contient très peu de paramètres et le reste reste gelé. Le changement du nombre total de paramètres s’explique par le remplacement de la tête de sortie vocabulaire (très grande) par une tête de classification binaire (beaucoup plus petite).

### **Question 4**

La loss diminue globalement au cours de l’entraînement avec quelques fluctuations ce qui indique que le modèle apprend correctement. La loss moyenne à la fin de l’époque est faible et on a une accuracy finale d’environ 92,79 % ce qui est élevé. Cette performance est raisonnable pour une tâche de classification de SMS spam/ham en utilisant un modèle de type GPT pré-entraîné avec une adaptation LoRA.

### **Question 5**

L’accuracy sur le jeu de test est d’environ 97,66 % ce qui est légèrement plus élevé que sur le train. Le modèle généralise bien et les performances sont cohérentes.


