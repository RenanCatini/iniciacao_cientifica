# Analise geral
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, precision_score
from imblearn.metrics import specificity_score
from matplotlib import pyplot as plt


def evaluate_model(y_true, y_pred, title="Matriz de Confusão"):
    """
    Avalia o desempenho de um modelo de classificação e exibe as métricas.

    Args:
        y_true (list, array): Os rótulos verdadeiros.
        y_pred (list, array): As previsões do modelo.
        title (str): O título para o gráfico da matriz de confusão.
    
    Returns:
        dict: Um dicionário contendo as métricas de avaliação.
    """

    cm = confusion_matrix(y_true, y_pred)

    # Exibir a matriz de confusão como gráfico
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "A"])
    disp.plot(cmap="Blues", values_format="d", text_kw={'size': 14})  
    plt.title(title)
    plt.show()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred), 
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred), 
        "specificity": specificity_score(y_true, y_pred)
    }

    # Imprimir métricas
    print("--- Métricas de Desempenho ---")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"ROC: {metrics['roc_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Specificity: {metrics['specificity']*100:.2f}%")
    print("------------------------------")
