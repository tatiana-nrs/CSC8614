# **rapport.md — TP5 Modèles de langage**
NIAURONIS Tatiana – FIPA 3A  
CSC8614 – TP5

---

## **Exercice 1 — Mise en place de TP5 et copie du RAG (base Chroma incluse)**

### **Question 1.a**

On a bien crée TP5 à la racine:

![alt text](image.png)

### **Question 1.b**

On a relancé Ollama sur le serveur.

### **Question 1.c**

On obtient:

![alt text](image-1.png)

---

## **Exercice 2 — Constituer un jeu de test (8–12 emails) pour piloter le développement**

### **Question 2.a/b**

On a crée le dossier avec les mails:

![alt text](image-2.png)

### **Question 2.c**

On a bien ajouté la section attendu.

### **Question 2.d**

Le jeu de test est composé de 9 emails stockés dans le dossier TP5/data/test_emails//
On a :

![alt text](image-3.png)

Le jeu de test couvre des cas réalistes et variés rencontrés dans un contexte d'un étudiant. Il inclut des emails purement informatifs, des messages nécessitant une réponse, des demandes ambiguës et des emails à risque impliquant des données personnelles sensibles. Cette diversité permet d’évaluer efficacement la capacité du système à détecter l’intention, le niveau de risque et l’action appropriée à entreprendre.

### **Question 2.f**

On obtient:

![alt text](image-4.png)

---

## **Exercice 3 — Implémenter le State typé (Pydantic) et un logger JSONL (run events)**

### **Question 3.b**

On a bien crée les dossiers:

![alt text](image-5.png)

### **Question 3.e**

Le fichier JSON est créé:

![alt text](image-6.png)

Et on affiche un extrait:

![alt text](image-7.png)

---

## **Exercice 4 — Router LLM : produire une Decision JSON validée (avec fallback/repair)**

### **Question 4.d**

La décision JSON affichée est:

![alt text](image-8.png)

Voici un extrait du run en JSONL:

```
{"run_id": "92f1b554-161c-4de2-8c60-0a2a68e655e3", "ts": "2026-01-23T16:38:42.267734Z", "event": "node_start", "data": {"node": "classify_email", "email_id": "E01"}}
{"run_id": "92f1b554-161c-4de2-8c60-0a2a68e655e3", "ts": "2026-01-23T16:39:24.590088Z", "event": "node_end", "data": {"node": "classify_email", "status": "ok", "decision": {"intent": "reply", "category": "admin", "priority": 1, "risk_level": "med", "needs_retrieval": false, "retrieval_query": "", "rationale": "Request for TOEIC registration with urgent administrative action required by the student."}}}
```

---

## **Exercice 5 — LangGraph : routing déterministe et graphe minimal (MVP)**

### **Question 5.a**

La commande utilisée est `pip install -U langgraph`. On a installé cette version:

![alt text](image-9.png)

### **Question 5.f**

À l'éxécution, on obtient:

![alt text](image-10.png)

On voit bien DECISION, DRAFT_V1 et ACTIONS.

Voici un extrait du fichier du run:

```
{"run_id": "00019f25-134a-4d3f-9d5c-b6f4241d26d1", "ts": "2026-01-23T16:57:51.139704Z", "event": "node_start", "data": {"node": "classify_email", "email_id": "E01"}}
{"run_id": "00019f25-134a-4d3f-9d5c-b6f4241d26d1", "ts": "2026-01-23T16:59:55.663374Z", "event": "node_end", "data": {"node": "classify_email", "status": "ok", "decision": {"intent": "reply", "category": "admin", "priority": 1, "risk_level": "med", "needs_retrieval": false, "retrieval_query": "", "rationale": "Demande administrative pour l'inscription au TOIEC le 25 octobre. L'action de l'étudiant est nécessaire."}}}
{"run_id": "00019f25-134a-4d3f-9d5c-b6f4241d26d1", "ts": "2026-01-23T16:59:55.704184Z", "event": "node_start", "data": {"node": "stub_reply"}}
{"run_id": "00019f25-134a-4d3f-9d5c-b6f4241d26d1", "ts": "2026-01-23T16:59:55.704850Z", "event": "node_end", "data": {"node": "stub_reply", "status": "ok"}}
```

---

## **Exercice 6 — Tool use : intégrer votre RAG comme outil (retrieval + evidence)**

### **Question 6.d**

A l'exécution, on a dans l'extrait JSONL:

```
{"run_id": "b8d70d67-ea09-4806-af69-dd28e00365ca", "ts": "2026-01-23T17:35:27.135418Z", "event": "tool_call", "data": {"tool": "rag_search", "args_hash": "a5832ef583df", "latency_ms": 4453, "status": "ok", "k": 5, "n_docs": 5}}
```

La présence d’evidence est confirmée par l’événement tool_call dans l'extrait JSONL, indiquant que l’outil rag_search a été appelé avec succès et a retourné n_docs = 5 ce qui prouve que des documents ont bien été récupérés.

---

## **Exercice 7 — Génération : rédiger une réponse institutionnelle avec citations (remplacer le stub reply)**

### **Question 7.c**

On a un ca reply avec evidence non vide:

![alt text](image-11.png)

![alt text](image-14.png)

Et un cas de safe mode:

![alt text](image-12.png)

![alt text](image-13.png)

---

## **Exercice 8 — Boucle contrôlée : réécriture de requête et 2e tentative de retrieval (max 2)**

### **Question 8.a**

On a bien modifié state.py:

![alt text](image-15.png)

### **Question 8.f**

On a bien 2 tentatives de retrieval comme on peut le voir ici:

![alt text](image-16.png)

Après une première tentative de retrieval ne retournant aucun document (n_docs = 0), le nœud draft_reply bascule en safe mode. Le signal last_draft_had_valid_citations = false déclenche alors une réécriture de la requête (rewrite_query) suivie d’une seconde tentative de retrieval.

L'extrait JSONL correspondant est:

```
{"event":"tool_call","data":{"tool":"rag_search","status":"ok","n_docs":0}}
{"event":"node_end","data":{"node":"draft_reply","status":"safe_mode","reason":"no_evidence"}}
{"event":"node_start","data":{"node":"rewrite_query"}}
{"event":"node_end","data":{"node":"rewrite_query","status":"ok","q2":"Montant CVEC remboursement expéditeur Jackie"}}
{"event":"tool_call","data":{"tool":"rag_search","status":"ok","n_docs":0}}
```

---

## **Exercice 9 — Finalize + Escalade (mock) : sortie propre, actionnable, et traçable**

### **Question 9.a**

On a bien modifié state.py:

![alt text](image-17.png)

### **Question 9.e**

On a testé deux mails dont 1 escalade.

Pour le reply on obtient:

![alt text](image-18.png)

On voit bien l'évènement finalize:

![alt text](image-19.png)

Pour le cas escalate, on obtient:

![alt text](image-20.png)

On observe bien l'action mockée handoff packet.
On voit aussi l'évènement finalize :

![alt text](image-21.png)

---

## **Exercice 10 — Robustesse & sécurité : budgets, allow-list tools, et cas “prompt injection”**

### **Question 10.d**

On voit bien que la décision est forcée en intent=escalate et que le risk_level est high. On voit aussi qu'un handoff_packet est produit:

![alt text](image-22.png)

Il n'y a pas d'appel rag_search dans les logs et on y voit injection_heuristic_triggered:

```
{"run_id": "317f5558-4240-48b3-b686-c7090744a31a", "ts": "2026-01-25T22:46:06.677655Z", "event": "node_start", "data": {"node": "classify_email", "email_id": "E13"}}
{"run_id": "317f5558-4240-48b3-b686-c7090744a31a", "ts": "2026-01-25T22:46:06.679351Z", "event": "node_end", "data": {"node": "classify_email", "status": "ok", "decision": {"intent": "escalate", "category": "other", "priority": 1, "risk_level": "high", "needs_retrieval": false, "retrieval_query": "", "rationale": "Suspicion de prompt injection."}, "note": "injection_heuristic_triggered"}}
{"run_id": "317f5558-4240-48b3-b686-c7090744a31a", "ts": "2026-01-25T22:46:06.679821Z", "event": "node_start", "data": {"node": "stub_escalate"}}
{"run_id": "317f5558-4240-48b3-b686-c7090744a31a", "ts": "2026-01-25T22:46:06.679870Z", "event": "node_end", "data": {"node": "stub_escalate", "status": "ok"}}
{"run_id": "317f5558-4240-48b3-b686-c7090744a31a", "ts": "2026-01-25T22:46:06.680088Z", "event": "node_start", "data": {"node": "finalize"}}
{"run_id": "317f5558-4240-48b3-b686-c7090744a31a", "ts": "2026-01-25T22:46:06.680137Z", "event": "node_end", "data": {"node": "finalize", "status": "ok", "final_kind": "handoff"}}
```

---

## **Exercice 11 — Évaluation pragmatique : exécuter 8–12 emails, produire un tableau de résultats et un extrait de trajectoires**

### **Question 11.a**

Le script s'est bien exécuté:

![alt text](image-24.png)

Il a produit un fichier batch_results.md:

![alt text](image-25.png)

### **Question 11.b**

### **Question 11.c**

Le tableau obtenu est:

| email_id | subject | intent | category | risk | final_kind | tool_calls | retrieval_attempts | notes |
|---|---|---|---|---|---|---:|---:|---|
| E01 | Inscription au TOIEC | reply | teaching | low | reply | 0 | 0 | run=f1a226bb-09cc-4fe5-9f08-46783c7d1960.jsonl |
| E02 | Urgent: Inscription administrative | reply | admin | med | reply | 1 | 1 | run=6776d6a3-19e7-4e35-a463-cb9ab08db308.jsonl |
| E03 | Publicité automatique | ignore | other | low | ignore | 0 | 0 | run=c02529d8-e29e-4705-b88b-62b5f0ecf669.jsonl |
| E04 | Problème urgent | reply | teaching | med | reply | 0 | 0 | run=a54e93d2-1d64-489e-89a8-509c3cf0b38a.jsonl |
| E05 | Erreur | reply | teaching | low | reply | 0 | 0 | run=927f661f-4325-4c8c-bd42-bf78235b6ea3.jsonl |
| E06 | Clarification à apporter | ask_clarification | other | med | clarification | 0 | 0 | run=03f4ef2f-86a8-40f6-a96a-40073dd4326f.jsonl |
| E07 | URGENT – Vérification de sécurité du compte utilisateur | escalate | other | high | handoff | 0 | 0 | run=5ed20563-c9ea-4d67-9a40-f52031441159.jsonl |
| E08 | Question concernant la validation du module CSC8607 | ask_clarification | teaching | med | clarification | 0 | 0 | run=7060e1e0-4bcb-4c6e-9f29-d2a4ab40ef92.jsonl |
| E09 | TOEIC 17 janvier 2026 : date limite d'inscription | reply | teaching | low | reply | 0 | 0 | run=a3f5ba69-ca0c-4539-ac37-5d1b2e2b0420.jsonl |
| E10 | Montant exact remboursement CVEC | reply | teaching | low | reply | 1 | 1 | run=9b361190-32de-474f-9e92-4376b264b500.jsonl |
| E11 | Contestation officielle – validation UE S7/ demande immédiat | ask_clarification | teaching | med | clarification | 0 | 0 | run=42f9e6ff-251d-4ae7-82ac-96b3319de89c.jsonl |
| E12 | Important | escalate | other | high | handoff | 0 | 0 | run=f05d8221-13bf-426a-84be-803baf1c064c.jsonl |

L’intent dominant est reply, représentant environ la moitié des emails testés.
On retrouve deux mails d'escalade, dont celui que nous avons rédigé à l'exercice précédent.
Les ask_clarification apparaissent lorsque l’email contient une demande incomplète ou ambigue.
La majorité des emails n’activent pas le RAG, ce qui est souhaitable pour limiter les appels inutiles. Les appels RAG sont réservés à des cas précis (TOEIC, CVEC) ici.

### **Question 11.d**

Run simple :E01 – Inscription au TOEIC:

L’email est d’abord analysé par le nœud classify_email, qui détecte un intent reply sans besoin de récupération documentaire (needs_retrieval=false).
Le nœud maybe_retrieve est donc ignoré et le nœud draft_reply génère directement une réponse. Cette trajectoire illustre un chemin court et optimal pour les demandes simples.

![alt text](image-26.png)

Run complexe : E02 – Inscription administrative urgente

Le nœud classify_email identifie une demande administrative nécessitant des informations vérifiables et active needs_retrieval=true. Le nœud maybe_retrieve effectue un appel au tool RAG (rag_search) et récupère plusieurs documents pertinents. Le nœud draft_reply génère une réponse contenant une citation issue des documents récupérés. Le nœud check_evidence valide la présence et la qualité des preuves dès la première tentative.

![alt text](image-27.png)


---

## **Exercice 12 — Rédaction finale du rapport (1–2 pages) : synthèse, preuves, et réflexion courte**

### **Question 12.a**

**Exécution**

**Réponse RAG sur un email**
python -m TP5.rag_answer

**Test unitaire du graphe minimal (1 email)**
python -m TP5.test_graph_minimal

**Évaluation batch sur le jeu de test**
python -m TP5.run_batch

Pour le run reply:

![alt text](image-18.png)

Pour le run escalate:

![alt text](image-20.png)

Ces captures montrent que l’agent produit une sortie finale (final_kind, final_text) et que le nœud finalize est systématiquement atteint.

### **Question 12.b**

**Architecture**

![alt text](image-29.png)

### **Question 12.c**

**Résultats**

![alt text](image-28.png)

### **Question 12.d**

**Trajectoire**

voir exercice précédent

### **Question 12.e**

**Ce qui marche bien:**

- La séparation claire des intents entre reply, ignore, ask_clarification et escalate.

- Le suivi des trajectoires (logs, budgets, evidence checks)  qui facilite le diagnostic et l’analyse des décisions de l’agent

**Ce qui est fragile:**

- Le routage initial dépend encore partiellement du modèle (biais vers reply qui est la valeur par défaut) et du prompt.

- Les intents ask_clarification, ignore et escalate sont moins roustes et systématiques que reply.

**Amélioration prioritaire (avec 2h de plus):**

- Renforcer ignore et ask_clarification à l’aide de règles déterministes simples (heuristiques lexicales) afin de réduire la dépendance au LLM et d’éviter les reply par défaut.























