# Trabalho Machine Learning – Algoritmo Naive Bayes

Este repositório contém o material do trabalho Machine Learning, focado no estudo e implementação do algoritmo **Naive Bayes** aplicado a problemas de **classificação**.

O projeto inclui:

- Uma POC simples com o dataset **Iris** (exemplo clássico de classificação em 3 classes).
- Duas POCs de **detecção de fraude em transações financeiras** usando dados sintéticos:
  - Versão **completa** (estatísticas + métricas principais).
  - Versão **avançada** (thresholds diferentes, AUC, análise de erros).

---

## 1. Estrutura do projeto

Arquivos principais:

- `naive_bayes_iris.py`  
  POC com o dataset Iris (`sklearn.datasets.load_iris`), classificando flores em três espécies.

- `naive_bayes_transacoes_completo.py`  
  POC de detecção de fraude usando um conjunto de dados sintético menor (`dados_transacoes_fraude_maior.csv`), com estatísticas descritivas, métricas de classificação e análise de erros (falsos positivos/negativos).

- `naive_bayes_transacoes_avancado.py`  
  Versão avançada da POC de fraude, usando um dataset maior (`dados_transacoes_fraude_5000.csv`) e avaliando o comportamento do modelo com diferentes **thresholds** de decisão, além de calcular **AUC (ROC)**.

- `dados_transacoes_fraude_maior.csv`  
  Conjunto de dados sintético de transações financeiras com rótulo de fraude (0 = não fraude, 1 = fraude), gerado com regras de negócio simples (valor alto, muitas tentativas, horário e distância de IP).

- `dados_transacoes_fraude_5000.csv`  
  Versão maior (≈ 5000 linhas) e mais equilibrada de dados de transações, usada na POC avançada.
---

## 2. Requisitos

- **Python 3.10+** (ou equivalente)
- Bibliotecas Python:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib` (usado na POC do Iris para exibir a matriz de confusão)

Para instalar tudo de uma vez:

```bash
pip install numpy pandas scikit-learn matplotlib
