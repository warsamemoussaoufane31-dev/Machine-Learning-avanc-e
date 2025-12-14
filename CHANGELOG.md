# Changelog

Toutes les modifications récentes (résumé clair des tâches effectuées) :

## Unreleased

- feat: add project scaffold (requirements, src, notebooks, CI, license)
  - Fichiers ajoutés : `requirements.txt`, `.gitignore`, `LICENSE`, `src/main.py`, `.github/workflows/ci.yml`, `notebooks/README.md`.
  - But : fournir un exemple minimal reproduisant le notebook Colab.

- feat: add `california-housing-ml` project scaffold
  - Dossier `california-housing-ml/` créé avec : `README.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.gitignore`, `src/` (modules pour pipeline ML), `notebooks/`, `models/`, `data/`, et workflow CI.
  - But : structure complète pour exécuter, entraîner et déployer un modèle ML.

- chore: add `tools/upload_notebook.py` and README note
  - Script CLI `tools/upload_notebook.py` ajouté pour copier un `.ipynb` local dans `california-housing-ml/notebooks/`, committer et pousser.
  - Mise à jour du `california-housing-ml/README.md` avec l'exemple d'utilisation.

- chore: import and convert Colab notebook
  - Le notebook fourni a été importé dans `california-housing-ml/notebooks/` et converti en script de travail (temporaire) pour faciliter l'intégration.

- chore: remove duplicate notebooks and unused converted script
  - Nettoyage des doublons et suppression des artefacts inutiles pour garder le dépôt propre.

- run: training executed locally (workspace)
  - `california-housing-ml/src/train_models.py` exécuté en environnement de test ; modèle sauvegardé : `california-housing-ml/models/first_pass_models.joblib` (meilleur : `forest_reg`).

Notes :
- Les messages de commit originaux restent inchangés (historique préservé). Ce `CHANGELOG.md` vise à donner des descriptions lisibles et centralisées pour faciliter la revue.
- Si tu souhaites une réécriture des messages de commit (alterner l'historique Git), indique-le explicitement (opération risquée).
