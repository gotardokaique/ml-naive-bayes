import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


def main():
    """
    POC - Naive Bayes aplicado a transações financeiras (fraude x não fraude).

    Espera um arquivo CSV na mesma pasta chamado:
        dados_transacoes_fraude.csv

    Colunas esperadas:
        - valor_transacao (R$)
        - num_tentativas_24h
        - hora_dia (0 a 23)
        - distancia_ip_km
        - fraude (0 = não, 1 = sim)
    """

    # 1. Carregar dados a partir do CSV
    df = pd.read_csv("dados_transacoes_fraude.csv")

    print("=== PRIMEIRAS LINHAS DO CONJUNTO DE DADOS ===")
    print(df.head())
    print()

    # 2. Estatísticas gerais
    print("=== ESTATÍSTICAS DESCRITIVAS GERAIS ===")
    print(df.describe())
    print()

    # 3. Distribuição da variável alvo (fraude)
    print("=== DISTRIBUIÇÃO DA VARIÁVEL ALVO (FRAUDE) ===")
    dist_alvo = df["fraude"].value_counts().rename(index={0: "não fraude", 1: "fraude"})
    print(dist_alvo)
    total = len(df)
    print()
    print("Proporção de fraude:")
    for idx, valor in dist_alvo.items():
        proporcao = valor / total
        print(f"  {idx}: {valor} registros ({proporcao:.2%})")
    print()

    # 4. Médias das features por classe
    print("=== MÉDIA DAS FEATURES POR CLASSE (FRAUDE x NÃO FRAUDE) ===")
    medias = df.groupby("fraude")[["valor_transacao", "num_tentativas_24h", "hora_dia", "distancia_ip_km"]].mean()
    medias = medias.rename(index={0: "não fraude", 1: "fraude"})
    print(medias)
    print()

    # 5. Separar features (X) e alvo (y)
    X = df[["valor_transacao", "num_tentativas_24h", "hora_dia", "distancia_ip_km"]]
    y = df["fraude"]

    # 6. Dividir em treino e teste
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

    # 7. Criar e treinar o modelo Naive Bayes
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    # 8. Fazer predições no conjunto de teste
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]  # probabilidade de fraude

    # 9. Avaliar desempenho
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, zero_division=0)
    revocacao = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=== RESULTADOS GERAIS DO MODELO (FRAUDE) ===")
    print(f"Acurácia : {acuracia:.2%}")
    print(f"Precisão : {precisao:.2%}")
    print(f"Revocação (Recall) : {revocacao:.2%}")
    print(f"F1-score : {f1:.2%}")
    print()
    print("Relatório de classificação detalhado:")
    print(classification_report(y_test, y_pred, target_names=["não fraude", "fraude"], zero_division=0))

    # 10. Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão (linhas = classe real, colunas = classe predita):")
    print(cm)
    print()
    print("Legenda da matriz de confusão:")
    print("  [0,0] = não fraude predita como não fraude (acerto)")
    print("  [0,1] = não fraude predita como fraude (falso positivo)")
    print("  [1,0] = fraude predita como não fraude (falso negativo)")
    print("  [1,1] = fraude predita como fraude (acerto)")
    print()

    # 11. Montar DataFrame de análise do conjunto de teste
    df_teste = X_test.copy()
    df_teste["fraude_real"] = y_test.values
    df_teste["fraude_predita"] = y_pred
    df_teste["prob_fraude_modelo"] = y_proba

    # Ordenar por maior probabilidade prevista de fraude
    df_teste_ordenado = df_teste.sort_values(by="prob_fraude_modelo", ascending=False)

    print("=== TOP 10 TRANSAÇÕES COM MAIOR PROBABILIDADE DE FRAUDE (SEGUNDO O MODELO) ===")
    print(df_teste_ordenado.head(10))
    print()

    # 12. Analisar erros mais críticos

    # Falsos positivos: modelo disse fraude (1), mas real é não fraude (0)
    falsos_positivos = df_teste[(df_teste["fraude_real"] == 0) & (df_teste["fraude_predita"] == 1)]
    # Falsos negativos: modelo disse não fraude (0), mas real é fraude (1)
    falsos_negativos = df_teste[(df_teste["fraude_real"] == 1) & (df_teste["fraude_predita"] == 0)]

    print("=== EXEMPLOS DE FALSOS POSITIVOS (NÃO FRAUDE MARCADA COMO FRAUDE) ===")
    if not falsos_positivos.empty:
        print(falsos_positivos.head(10))
    else:
        print("Nenhum falso positivo encontrado nas primeiras amostras.")
    print()

    print("=== EXEMPLOS DE FALSOS NEGATIVOS (FRAUDE NÃO DETECTADA) ===")
    if not falsos_negativos.empty:
        print(falsos_negativos.head(10))
    else:
        print("Nenhum falso negativo encontrado nas primeiras amostras.")
    print()


if __name__ == "__main__":
    main()
