# Git Survival Analysis

C'est un petit projet que j'ai monté pour appliquer des concepts d'analyse de survie aux historiques Git. L'objectif est d'essayer de comprendre pourquoi et quand les développeurs arrêtent de contribuer à un projet.

## C'est quoi l'analyse de survie ?

À la base, ce sont des méthodes mathématiques et statistiques plutôt utilisées en médecine (pour estimer le temps avant la rémission ou le décès d'un patient) ou en ingénierie (le temps avant qu'une pièce mécanique casse). 

Ici, le "patient" c'est un développeur, et "l'événement" (le décès) c'est le moment où il abandonne le projet (le churn). La difficulté de ce genre d'étude, c'est la "censure" : si un contributeur est arrivé le mois dernier et n'a pas encore recomitté, on ne peut pas dire qu'il a abandonné (il est juste trop récent). L'analyse de survie gère ces données incomplètes proprement.

## Les modèles statistiques utilisés

L'outil se base sur la librairie `lifelines` et implémente 3 méthodes différentes :

1. **Kaplan-Meier :** C'est une courbe très visuelle qui donne la probabilité globale de rétention au fil du temps. Par exemple : "après 100 jours, 60% des développeurs contribuent encore".
2. **Modèle de Cox (Proportional Hazards) :** C'est un modèle de régression multivariée. Ça permet d'isoler l'impact de certaines habitudes sur le risque de départ. Le modèle renvoie des "Hazard Ratios" pour chaque variable. J'analyse pas mal de trucs :
   - Horaires (ratio de commits la nuit/le week-end)
   - "Bus Factor" (est-ce qu'il est seul à toucher ses fichiers ?)
   - NLP / Sentiment (analyse des messages de commit avec TextBlob)
   - Contagion (est-ce que ses collègues proches ont déjà abandonné le projet ?)
3. **Modèle AFT (Accelerated Failure Time) :** Contrairement à Cox qui étudie le "risque", l'AFT modélise directement le temps qu'il reste. J'ai configuré l'outil pour qu'il teste plusieurs lois de probabilité (Weibull, Log-Normal) et choisisse la meilleure.

L'outil intègre aussi un petit algorithme de détection des départs imminents (pour lister les contributeurs actifs actuels qui ont le plus gros risque de bientôt décrocher).

## Comment ça s'utilise

Il faut installer les dépendances avec Poetry (pandas, pydriller, lifelines, plotly...) et ensuite lancer le script sur un repo local ou une URL :

```bash
poetry run python -m cli /chemin/vers/le/repo --output rapport.html
```

J'ai ajouté un système de cache pour éviter de re-parser tout l'historique si on relance l'outil deux fois d'affilée. Le rapport généré est une page HTML standalone avec des graphiques interactifs (Plotly).

J'espère que ce projet pourra servir ! N'hésitez pas à jeter un œil au code, tout est dans le dossier `src/`.
