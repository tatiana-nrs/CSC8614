# Re: [PRO 8605] MAIA 3A - Anomaly Detection

**From:** Alexandre LAURET <alexandre.lauret@telecom-sudparis.eu>

**Date:** Thu, 15 Jan 2026 11:48:24 +0100

**Message-ID:** <250243398.32682418.1768474104303.JavaMail.zimbra@telecom-sudparis.eu>

---

Bonjour Monsieur Romero, 


Voici les descriptions demandées pour le projet. 

Fonctionnalités 
Ce projet développe un outil de détection d'erreurs destiné aux chaînes de production des industries manufacturières. Il utilise des images de produits, des historiques de production et des dessins industriels pour alimenter des modèles d'intelligence artificielle. Le système combine la classification d'images et la détection d'anomalies sur des images en deux dimensions, mais l'objectif à terme est d'adapter un réseau généraliste capable d'apprendre sur ces données 2D pour appliquer ensuite ses analyses sur des modèles en 3D. 

État Intermédiaire 
Pour la soutenance de mi-parcours, nous visons d'avoir une chaîne de traitement fonctionnel sur des images de 2D. Ceci avec une API fonctionnelle pouvant proposer des prédictions. En plus de cette API, un site web faciliterait l'utilisation au client occasionnel. 

Planning Prévisionnel 
Semaine Catégorie Tâche Objectif Concret pour le 6 Février S1 : 15-22 Jan 	IA & Data 	Finalisation du Dataset 2D 	Constituer et nettoyer le jeu d'images 2D (produits conformes/non-conformes) pour la démo. ​ 
S1 : 15-22 Jan 	IA & Data 	Entraînement Modèle 2D 	Avoir un modèle (classification ou anomalie) fonctionnel qui tourne et prédit correctement sur les images de test. 
S2 : 23-30 Jan 	Backend 	Développement API 	Mettre en place l'API avec FastAPI et un endpoint qui reçoit une image et renvoie le résultat de l'IA. ​ 
S2 : 23-30 Jan 	Frontend 	Interface Web Minimaliste 	Créer une page simple : un bouton "Upload" et une zone d'affichage pour l'image et le résultat (sans design complexe). 
S3 : 30 Jan - 5 Fév 	Intégration 	Liaison Site-API-IA 	Connecter le site à l'API pour que l'analyse se fasse en temps réel lors de l'upload d'une image. ​ 
S3 : 30 Jan - 5 Fév 	Soutenance 	Préparation Présentation 	Rédiger les slides incluant l'état d'avancement et la perspective future (réseau généraliste 2D vers 3D). ​ 
6 Février 	Jalon 	Soutenance Mi-Parcours 	Présentation + Démo : Upload d'une image 2D, Analyse par l'IA et Affichage du résultat. 


Cordialement, 
Le Groupe Tatiana, Salim, Alexandre 




De: "LAURET Alexandre" <alexandre.lauret@telecom-sudparis.eu> 
À: "ROMERO Julien" <julien.romero@telecom-sudparis.eu> 
Cc: "NIAURONIS Tatiana" <tatiana.niauronis@telecom-sudparis.eu>, "salim jerbi" <salim.jerbi@telecom-sudparis.eu> 
Envoyé: Jeudi 8 Janvier 2026 19:08:36 
Objet: [PRO 8605] MAIA 3A - Anomaly Detection 

Bonjour Monsieur Romero, 


Notre groupe, composé de Tatiana Niauronis, Salim Jerbi et moi-même (Alexandre Lauret), avons choisi le projet suivant : 

Outil de détection d'erreurs sur chaînes de production 

Données collectées : Images des produits, historiques de production, dessins industriels. 
Datasets : À créer si besoin. 
Clients potentiels : Industries manufacturières. 
Modèles : Classification d'images (ML) et détection d’anomalies (DL). 

Concernant les bases de données utilisées, nous commencerons avec le MVTEC Anomaly Detection, rendre le modèle assez généraliste pour continuer à l'entraîner sur le MVTEC Anomaly Detection 2 et si possible le fine tuner pour le rendre utilisable sur de la 3D. 
Ainsi, notre projet pourra se conformer à n'importe quelle chaîne de production. 


Cordialement, 
Le Groupe Tatiana, Salim, Alexandre
