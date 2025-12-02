import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def imprimir_matriz_confusao(cm: "np.ndarray"):
    print("Matriz de confusão (linhas = classe real, colunas = classe predita):")
    print(cm)
    print("Legenda:")
    print("  [0,0] = não fraude predita como não fraude (acerto)")
    print("  [0,1] = não fraude predita como fraude (falso positivo)")
    print("  [1,0] = fraude predita como não fraude (falso negativo)")
    print("  [1,1] = fraude predita como fraude (acerto)")
    print()


def avaliar_com_threshold(y_true, prob_fraude, threshold: float, label: str):
    from numpy import array

    y_true = array(y_true)
    y_pred = (prob_fraude >= threshold).astype(int)

    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(y_true, y_pred, zero_division=0)
    revocacao = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"=== RESULTADOS DO MODELO ({label}) ===")
    print(f"Threshold usado para fraude: {threshold:.2f}")
    print(f"Acurácia : {acuracia:.2%}")
    print(f"Precisão : {precisao:.2%}")
    print(f"Revocação (Recall) : {revocacao:.2%}")
    print(f"F1-score : {f1:.2%}")
    print()
    print("Relatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=['não fraude', 'fraude'], zero_division=0))
    imprimir_matriz_confusao(cm)

    return y_pred, cm, {
        "acuracia": acuracia,
        "precisao": precisao,
        "revocacao": revocacao,
        "f1": f1,
    }


def main():
    """

    Espera um arquivo CSV na mesma pasta chamado:
        dados_transacoes_fraude_5000.csv

    Colunas esperadas:
        - valor_transacao (R$)
        - num_tentativas_24h
        - hora_dia (0 a 23)
        - distancia_ip_km
        - fraude (0 = não, 1 = sim)
    """

    # 1. Carregar dados a partir do CSV
    df = pd.read_csv("dados_transacoes_fraude_5000.csv")

    print("=== PRIMEIRAS LINHAS DO CONJUNTO DE DADOS ===")
    print(df.head())
    print()

    print("=== ESTATÍSTICAS DESCRITIVAS GERAIS ===")
    print(df.describe())
    print()

    print("=== DISTRIBUIÇÃO DA VARIÁVEL ALVO (FRAUDE) ===")
    dist_alvo = df["fraude"].value_counts().rename(index={0: "não fraude", 1: "fraude"})
    print(dist_alvo)
    total = len(df)
    print()
    print("Proporção de cada classe:")
    for idx, valor in dist_alvo.items():
        proporcao = valor / total
        print(f"  {idx}: {valor} registros ({proporcao:.2%})")
    print()

    print("=== MÉDIA DAS FEATURES POR CLASSE (FRAUDE x NÃO FRAUDE) ===")
    medias = df.groupby("fraude")[["valor_transacao", "num_tentativas_24h", "hora_dia", "distancia_ip_km"]].mean()
    medias = medias.rename(index={0: "não fraude", 1: "fraude"})
    print(medias)
    print()

    # 2. Separar features (X) e alvo (y)
    X = df[["valor_transacao", "num_tentativas_24h", "hora_dia", "distancia_ip_km"]]
    y = df["fraude"]

    # 3. Dividir em treino e teste
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

    # 4. Criar e treinar o modelo Naive Bayes
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    # 5. Probabilidades previstas de fraude
    prob_fraude = modelo.predict_proba(X_test)[:, 1]

    # 6. Avaliar com threshold padrão (0.5)
    y_pred_05, cm_05, metrics_05 = avaliar_com_threshold(
        y_true=y_test,
        prob_fraude=prob_fraude,
        threshold=0.5,
        label="Threshold padrão (0.50)",
    )

    # 7. Avaliar com threshold mais conservador (0.8)
    y_pred_08, cm_08, metrics_08 = avaliar_com_threshold(
        y_true=y_test,
        prob_fraude=prob_fraude,
        threshold=0.8,
        label="Threshold conservador (0.80)",
    )

    # 8. AUC (avaliando quão bem o modelo ordena fraude x não fraude)
    try:
        auc = roc_auc_score(y_test, prob_fraude)
        print("=== MÉTRICA ADICIONAL ===")
        print(f"AUC (Área sob a curva ROC): {auc:.3f}")
        print("Quanto mais próximo de 1, melhor a separação entre fraude e não fraude.")
        print()
    except Exception as e:
        print("Não foi possível calcular AUC:", e)

    # 9. Construir DataFrame de análise com o threshold padrão (0.5)
    df_teste = X_test.copy()
    df_teste["fraude_real"] = y_test.values
    df_teste["prob_fraude_modelo"] = prob_fraude
    df_teste["fraude_predita_05"] = y_pred_05
    df_teste["fraude_predita_08"] = y_pred_08

    df_teste_ordenado = df_teste.sort_values(by="prob_fraude_modelo", ascending=False)

    print("=== TOP 10 TRANSAÇÕES COM MAIOR PROBABILIDADE DE FRAUDE ===")
    print(df_teste_ordenado.head(10))
    print()

    falsos_positivos_05 = df_teste[
        (df_teste["fraude_real"] == 0) & (df_teste["fraude_predita_05"] == 1)
    ]
    falsos_negativos_05 = df_teste[
        (df_teste["fraude_real"] == 1) & (df_teste["fraude_predita_05"] == 0)
    ]

    print("=== FALSOS POSITIVOS (THRESHOLD 0.50) ===")
    if not falsos_positivos_05.empty:
        print(falsos_positivos_05.head(10))
    else:
        print("Nenhum falso positivo encontrado para threshold 0.50.")
    print()

    print("=== FALSOS NEGATIVOS (THRESHOLD 0.50) ===")
    if not falsos_negativos_05.empty:
        print(falsos_negativos_05.head(10))
    else:
        print("Nenhum falso negativo encontrado para threshold 0.50.")
    print()


if __name__ == "__main__":
    main()
