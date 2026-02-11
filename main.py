# Declaração de uso de IAGen (CONPEP/UFOP)
# Este arquivo/trecho teve suporte assistivo de IAGen para geração/refatoração de código:
# Ferramenta: Claude Sonnet 4.5 (Thinking) (via IDE)
# Data aproximada: 05/02/2026 – 09/02/2026
# Observação: todo output foi revisado, testado e adaptado.

import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("CARREGANDO DATASET")
print("=" * 80)

df = pd.read_csv('dataset.csv')

print(f"Dataset carregado com sucesso!")
print(f"Número total de registros: {len(df)}")
print(f"Colunas disponíveis: {list(df.columns)}")
print(f"\nPrimeiras linhas do dataset:")
print(df.head())
print(f"\nDistribuição da variável alvo (viral):")
print(df['viral'].value_counts())
print()

# ============================================================================
# 3. SEPARAR FEATURES (X) E TARGET (y)
# ============================================================================
print("=" * 80)
print("SEPARANDO FEATURES E TARGET")
print("=" * 80)

X = df['descricao_visual']
y = df['viral']

print(f"Features (X): {X.shape[0]} amostras")
print(f"Target (y): {y.shape[0]} amostras")
print(f"Proporção de classes em y:")
print(f"  - Não viral (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
print(f"  - Viral (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
print()

# ============================================================================
# 3.1. LIMPEZA DE DADOS (REMOVER VALORES NULOS)
# ============================================================================
print("=" * 80)
print("LIMPEZA DE DADOS")
print("=" * 80)

nulos_descricao = X.isna().sum()
nulos_viral = y.isna().sum()

print(f"Valores nulos encontrados:")
print(f"  - descricao_visual: {nulos_descricao}")
print(f"  - viral: {nulos_viral}")

df_limpo = pd.DataFrame({'descricao_visual': X, 'viral': y})

# Remover linhas com valores nulos em qualquer coluna
tamanho_antes = len(df_limpo)
df_limpo = df_limpo.dropna()
tamanho_depois = len(df_limpo)

print(f"\nRegistros removidos: {tamanho_antes - tamanho_depois}")
print(f"Registros restantes: {tamanho_depois}")

# Atualizar X e y com dados limpos
X = df_limpo['descricao_visual']
y = df_limpo['viral']

# Garantir que y seja do tipo int (em caso de conversão automática para float)
y = y.astype(int)

print(f"\nDados após limpeza:")
print(f"  - Features (X): {X.shape[0]} amostras")
print(f"  - Target (y): {y.shape[0]} amostras")
print(f"  - Proporção de classes:")
print(f"    • Não viral (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
print(f"    • Viral (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
print()

# ============================================================================
# 4. DIVIDIR DADOS EM TREINO (80%) E TESTE (20%)
# ============================================================================
print("=" * 80)
print("DIVISÃO DOS DADOS EM TREINO E TESTE")
print("=" * 80)

# Dividir com stratify para manter a proporção de classes
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,      # 20% para teste
    stratify=y,          # Manter proporção de classes
    random_state=42      # Reprodutibilidade
)

print(f"Conjunto de TREINO: {len(X_train)} amostras ({len(X_train) / len(X) * 100:.1f}%)")
print(f"  - Não viral (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
print(f"  - Viral (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")
print()
print(f"Conjunto de TESTE: {len(X_test)} amostras ({len(X_test) / len(X) * 100:.1f}%)")
print(f"  - Não viral (0): {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
print(f"  - Viral (1): {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")
print()

# ============================================================================
# 5. CRIAR O TFIDF VECTORIZER
# ============================================================================
print("=" * 80)
print("CRIAÇÃO DO TF-IDF VECTORIZER")
print("=" * 80)

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,      # Limitar a 5000 features mais importantes
    ngram_range=(1, 2),     # Usar unigramas e bigramas
    min_df=2,               # Ignorar termos que aparecem em menos de 2 documentos
    max_df=0.95,            # Ignorar termos que aparecem em mais de 95% dos documentos
    strip_accents='unicode' # Remover acentos
)

print("TF-IDF Vectorizer criado com os seguintes parâmetros:")
print(f"  - max_features: 5000")
print(f"  - ngram_range: (1, 2)")
print(f"  - min_df: 2")
print(f"  - max_df: 0.95")
print()

# ============================================================================
# 6. FIT E TRANSFORM NO TREINO, APENAS TRANSFORM NO TESTE
# ============================================================================
print("=" * 80)
print("VETORIZAÇÃO TF-IDF (SEM VAZAMENTO DE DADOS)")
print("=" * 80)

# Fazer fit() APENAS no conjunto de treino
# Isso evita vazamento de dados (data leakage)
print("Aplicando fit() no conjunto de TREINO...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("Aplicando transform() no conjunto de TESTE...")

X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nMatriz TF-IDF de TREINO: {X_train_tfidf.shape} (amostras x features)")
print(f"Matriz TF-IDF de TESTE: {X_test_tfidf.shape} (amostras x features)")
print(f"Vocabulário aprendido: {len(tfidf_vectorizer.vocabulary_)} termos únicos")
print()

# ============================================================================
# 7. CRIAR MODELO DE REGRESSÃO LOGÍSTICA
# ============================================================================
print("=" * 80)
print("CRIAÇÃO DO MODELO DE REGRESSÃO LOGÍSTICA")
print("=" * 80)

modelo = LogisticRegression(
    max_iter=1000,          # Número máximo de iterações
    random_state=42,        # Reprodutibilidade
    solver='lbfgs',         # Algoritmo de otimização
    class_weight='balanced' # Balancear classes automaticamente
)

print("Modelo de Regressão Logística criado com os seguintes parâmetros:")
print(f"  - max_iter: 1000")
print(f"  - solver: lbfgs")
print(f"  - class_weight: balanced")
print()

# ============================================================================
# 8. TREINAR O MODELO (MEDIR TEMPO DE TREINO)
# ============================================================================
print("=" * 80)
print("TREINAMENTO DO MODELO")
print("=" * 80)

inicio_treino = time.time()

modelo.fit(X_train_tfidf, y_train)

tempo_treino = time.time() - inicio_treino

print(f"✓ Modelo treinado com sucesso!")
print(f"✓ Tempo de treino: {tempo_treino:.4f} segundos")
print()

# ============================================================================
# 9. FAZER PREDIÇÕES NO CONJUNTO DE TESTE (MEDIR TEMPO DE PREDIÇÃO)
# ============================================================================
print("=" * 80)
print("PREDIÇÃO NO CONJUNTO DE TESTE")
print("=" * 80)

inicio_predicao = time.time()

y_pred = modelo.predict(X_test_tfidf)

tempo_predicao = time.time() - inicio_predicao

print(f"✓ Predições realizadas com sucesso!")
print(f"✓ Tempo de predição: {tempo_predicao:.4f} segundos")
print()

# ============================================================================
# 9.1. SALVAR MODELO E VETORIZADOR TF-IDF
# ============================================================================
print("=" * 80)
print("SALVANDO MODELO E VETORIZADOR TF-IDF")
print("=" * 80)

nome_arquivo_modelo = 'modelo_viralidade.pkl'
nome_arquivo_tfidf = 'vetor_tfidf.pkl'

joblib.dump(modelo, nome_arquivo_modelo)
print(f"✓ Modelo de Regressão Logística salvo com sucesso em: {nome_arquivo_modelo}")

joblib.dump(tfidf_vectorizer, nome_arquivo_tfidf)
print(f"✓ Vetorizador TF-IDF salvo com sucesso em: {nome_arquivo_tfidf}")
print()
print("Os arquivos podem ser carregados posteriormente utilizando:")
print(f"  modelo = joblib.load('{nome_arquivo_modelo}')")
print(f"  tfidf_vectorizer = joblib.load('{nome_arquivo_tfidf}')")
print()

# ============================================================================
# 10. CALCULAR MÉTRICAS DE AVALIAÇÃO
# ============================================================================
print("=" * 80)
print("MÉTRICAS DE AVALIAÇÃO DO MODELO")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Exibir métricas
print(f"ACCURACY (Acurácia):   {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"PRECISION (Precisão):  {precision:.4f} ({precision * 100:.2f}%)")
print(f"RECALL (Revocação):    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-SCORE:              {f1:.4f} ({f1 * 100:.2f}%)")
print()

# ============================================================================
# 11. EXIBIR MATRIZ DE CONFUSÃO
# ============================================================================
print("=" * 80)
print("MATRIZ DE CONFUSÃO")
print("=" * 80)

# Calcular matriz de confusão
cm = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão:")
print()
print("                  Predito")
print("                 0      1")
print("           ┌─────────────┐")
print(f"Real    0  │ {cm[0][0]:4d}  {cm[0][1]:4d} │")
print(f"        1  │ {cm[1][0]:4d}  {cm[1][1]:4d} │")
print("           └─────────────┘")
print()
print("Legenda:")
print(f"  - Verdadeiros Negativos (TN): {cm[0][0]} (corretamente previsto como não viral)")
print(f"  - Falsos Positivos (FP):      {cm[0][1]} (incorretamente previsto como viral)")
print(f"  - Falsos Negativos (FN):      {cm[1][0]} (incorretamente previsto como não viral)")
print(f"  - Verdadeiros Positivos (TP): {cm[1][1]} (corretamente previsto como viral)")
print()

# ============================================================================
# 12. RESUMO FINAL COM TEMPOS DE EXECUÇÃO
# ============================================================================
print("=" * 80)
print("RESUMO FINAL")
print("=" * 80)
print()
print("📊 MÉTRICAS DE DESEMPENHO:")
print(f"   • Accuracy:   {accuracy:.4f}")
print(f"   • Precision:  {precision:.4f}")
print(f"   • Recall:     {recall:.4f}")
print(f"   • F1-Score:   {f1:.4f}")
print()
print("⏱️  TEMPOS DE EXECUÇÃO:")
print(f"   • Tempo de treino:    {tempo_treino:.4f} segundos")
print(f"   • Tempo de predição:  {tempo_predicao:.4f} segundos")
print()
print("📈 CONJUNTO DE DADOS:")
print(f"   • Total de amostras:  {len(df)}")
print(f"   • Treino:             {len(X_train)} amostras (80%)")
print(f"   • Teste:              {len(X_test)} amostras (20%)")
print()
print("=" * 80)
print("PIPELINE CONCLUÍDO COM SUCESSO! ✓")
print("=" * 80)
