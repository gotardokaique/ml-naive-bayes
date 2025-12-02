import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    """
    POC - Classificação com Naive Bayes usando o dataset Iris.

    Fluxo:
    1. Carregar o dataset Iris.
    2. Separar em treino e teste.
    3. Treinar o modelo Gaussian Naive Bayes.
    4. Avaliar (acurácia + relatório de classificação).
    5. Exibir matriz de confusão em texto e gráfico.
    """

    # 1. Carregar o dataset Iris
    iris = load_iris()
    X = iris.data              # características (features)
    y = iris.target            # rótulos (classes)
    target_names = iris.target_names

    # Contagem de amostras por classe
    class_counts = np.bincount(y)

    print("=== INFORMAÇÕES DO CONJUNTO DE DADOS (IRIS) ===")
    print(f"Número total de amostras : {X.shape[0]}")
    print(f"Número de atributos      : {X.shape[1]}")
    print("Classes disponíveis      :")
    for idx, name in enumerate(target_names):
        print(f"  - {idx}: {name} ({class_counts[idx]} amostras)")
    print()

    # 2. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("=== DIVISÃO TREINO/TESTE ===")
    print(f"Tamanho do conjunto de treino : {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste  : {X_test.shape[0]} amostras")
    print()

    # 3. Criar e treinar o modelo Naive Bayes (gaussiano)
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    # 4. Fazer predições no conjunto de teste
    y_pred = modelo.predict(X_test)

    # 5. Avaliar o desempenho
    acuracia = accuracy_score(y_test, y_pred)

    print("=== RESULTADOS DO MODELO NAIVE BAYES ===")
    print(f"Acurácia no conjunto de teste: {acuracia:.2%}")  # em porcentagem
    print()
    print("Relatório de classificação (por classe):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão (linhas = classe real, colunas = classe predita):")
    print(cm)
    print()

    # 7. Exibir matriz de confusão em gráfico
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_title("Matriz de Confusão - Naive Bayes (Iris)")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Escrever os valores dentro das células
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
