# **rapport.md — TP2 csc8614-language-models**
NIAURONIS Tatiana – FIPA 3A  
CSC8614 – TP2

---

### **Question 1**

```bash
conda create -n csc8614_tp2 python=3.10
conda activate csc8614_tp2
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
  
### **Question 2**

settings est un dictionnaire Python (dict) qui contient les hyperparamètres du modèle GPT-2.
Il comprend 5 clés qui sont n_vocab, n_ctx, n_embd, n_head et n_layer et décrivent la taille du vocabulaire, la longueur maximale du contexte, la dimension des embeddings, le nombre de têtes d’attention et le nombre de couches Transformer. Ces paramètres définissent l’architecture du modèle.

### **Question 3**

params est un dictionnaire Python (dict) qui contient les poids du modèle GPT-2. L’une des clés principales est blocks qui est une liste. Chaque élément de cette liste est un dictionnaire contenant les tenseurs PyTorch des paramètres internes d’un bloc Transformer (attention, MLP, normalisation).

### **Question 4**

Le constructeur __init__ attend en entrée un dictionnaire de configuration cfg contenant des clés spécifiques au modèle interne. Ces clés sont notamment vocab_size, context_length, emb_dim, n_heads, n_layers ainsi que des paramètres supplémentaires comme drop_rate et qkv_bias.

La variable settings est un dictionnaire, mais elle utilise des noms de clés différents et n’est donc pas directement compatible avec la structure attendue par GPTModel. Il est nécessaire d’effectuer un mapping entre les clés de settings et celles attendues par le modèle.

### **Question 5.1**

On le fait pour mélanger (shuffle) aléatoirement toutes les lignes du dataset avant de le découper en train/test. Sinon, si les données sont dans un certain ordre, le split 80/20 pourrait créer un train et un test pas représentatifs. Le random_state=123 sert à rendre ce mélange reproductible et à chaque exécution, on obtient le même shuffle donc le même train/test split.

### **Question 5.2**

Dans le jeu d’entraînement, on a 3860 messages ham et 597 messages spam donc les classes sont déséquilibrées avec une forte majorité de messages ham par rapport aux messages spam.

Ce déséquilibre peut poser problème lors du fine-tuning du modèle car le modèle peut apprendre à privilégier la classe majoritaire (ham) et obtenir une bonne accuracy globale tout en étant moins performant pour détecter la classe minoritaire (spam). 

### **Question 8.3**

On gèle les couches internes afin de conserver les représentations linguistiques pré-entraînées de GPT-2 et d’entraîner uniquement la tête de classification. Cela réduit le coût de calcul et limite le sur-apprentissage sur un petit jeu de données.

### **Question 10**

Au sein d’une époque la loss bouge beaucoup (de 0.5 à 0.96). C’est normal en mini-batch donc la courbe n’est pas parfaitement décroissante batch par batch.

Pour l'epoch 1, on a  Train Acc ≈ 86.47%, mais Spam Acc = 0% (train et test). Le modèle a surtout appris la classe majoritaire (“ham”) et prédit quasiment tout en ham.

On a un changement à l'epoch 2 car Train Acc 88.22%, Test Acc 90.58% et surtout Spam Acc ≈ 81%. Le modèle commence à apprendre à détecter le spam.

À l'epoch 3,  Spam Acc monte encore (≈ 90–93%), mais accuracy globale baisse (Train 81.40%, Test 83.59%) car on a plus de spams détectés, mais il fait aussi plus de faux positifs.

### **Question 11**

En augmentant le learning rate à 1e-4 et en limitant l’entraînement à 2 epochs, le modèle présente un comportement plus stable. La loss diminue régulièrement et l’accuracy sur le jeu de test atteint environ 89.7%, tandis que la détection du spam progresse sans que le modèle ne prédise systématiquement la classe minoritaire.

### **Question 12**

Le modèle arrive à séparer correctement et il peut reconnaître certains motifs évidents (gagné, cadeau, lien).