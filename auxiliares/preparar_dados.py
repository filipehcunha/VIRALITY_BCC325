# Declaração de uso de IAGen (CONPEP/UFOP)
# Este arquivo/trecho teve suporte assistivo de IAGen para geração/refatoração de código:
# Ferramenta: Claude Sonnet 4.5 (Thinking) (via IDE)
# Data aproximada: 05/02/2026 – 09/02/2026
# Observação: todo output foi revisado, testado e adaptado.


import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# 1. CARREGAR O DATASET
# ============================================
print("="*60)
print("PREPARACAO DO DATASET DE VIRALIDADE")
print("="*60)

# Caminho para o arquivo CSV
caminho_csv = 'resultados/dataset_taxa_engajamento.csv'

# Carregar o dataset
print(f"\nCarregando dataset: {caminho_csv}")
df = pd.read_csv(caminho_csv)

print(f"Dataset carregado com sucesso!")
print(f"  Total de registros: {len(df)}")
print(f"  Colunas: {list(df.columns)}")

# ============================================
# 2. SEPARAR FEATURES (X) E TARGET (y)
# ============================================
print("\n" + "="*60)
print("SEPARACAO DE FEATURES E TARGET")
print("="*60)

# X = feature de texto (Descricao Visual)
X = df['Descrição Visual']

# y = target binario (viral: 0 ou 1)
y = df['viral']

print(f"Features (X): coluna 'Descricao Visual'")
print(f"  Tipo: {X.dtype}")
print(f"  Amostras: {len(X)}")

print(f"\nTarget (y): coluna 'viral'")
print(f"  Tipo: {y.dtype}")
print(f"  Amostras: {len(y)}")

# ============================================
# 3. VERIFICAR DISTRIBUICAO INICIAL
# ============================================
print("\n" + "="*60)
print("DISTRIBUICAO DA CLASSE TARGET (ANTES DA DIVISAO)")
print("="*60)

# Contagem absoluta
contagem_total = y.value_counts().sort_index()
print("\nContagem absoluta:")
print(f"  Nao viral (0): {contagem_total[0]} registros")
print(f"  Viral (1): {contagem_total[1]} registros")

# Proporcao percentual
proporcao_total = y.value_counts(normalize=True).sort_index() * 100
print("\nProporcao percentual:")
print(f"  Nao viral (0): {proporcao_total[0]:.2f}%")
print(f"  Viral (1): {proporcao_total[1]:.2f}%")

# ============================================
# 4. DIVISAO ESTRATIFICADA (80% TREINO, 20% TESTE)
# ============================================
print("\n" + "="*60)
print("DIVISAO ESTRATIFICADA DO DATASET")
print("="*60)

# Parametros da divisao:
# - test_size=0.2 -> 20% para teste, 80% para treino
# - stratify=y -> mantem a mesma proporcao de classes em treino e teste
# - random_state=42 -> garante reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"Divisao realizada com sucesso!")
print(f"\nConfiguracao:")
print(f"  Tamanho do teste: 20%")
print(f"  Tamanho do treino: 80%")
print(f"  Estratificacao: Ativada")
print(f"  Random state: 42")

# ============================================
# 5. TAMANHO DOS CONJUNTOS
# ============================================
print("\n" + "="*60)
print("TAMANHO DOS CONJUNTOS")
print("="*60)

print(f"\nConjunto de TREINO:")
print(f"  X_train: {len(X_train)} amostras")
print(f"  y_train: {len(y_train)} amostras")

print(f"\nConjunto de TESTE:")
print(f"  X_test: {len(X_test)} amostras")
print(f"  y_test: {len(y_test)} amostras")

print(f"\nProporcao:")
print(f"  Treino: {len(X_train)/len(X)*100:.1f}%")
print(f"  Teste: {len(X_test)/len(X)*100:.1f}%")

# ============================================
# 6. DISTRIBUICAO DAS CLASSES NO TREINO
# ============================================
print("\n" + "="*60)
print("DISTRIBUICAO NO CONJUNTO DE TREINO")
print("="*60)

# Contagem absoluta no treino
contagem_treino = y_train.value_counts().sort_index()
print("\nContagem absoluta:")
print(f"  Nao viral (0): {contagem_treino[0]} registros")
print(f"  Viral (1): {contagem_treino[1]} registros")

# Proporcao percentual no treino
proporcao_treino = y_train.value_counts(normalize=True).sort_index() * 100
print("\nProporcao percentual:")
print(f"  Nao viral (0): {proporcao_treino[0]:.2f}%")
print(f"  Viral (1): {proporcao_treino[1]:.2f}%")

# ============================================
# 7. DISTRIBUICAO DAS CLASSES NO TESTE
# ============================================
print("\n" + "="*60)
print("DISTRIBUICAO NO CONJUNTO DE TESTE")
print("="*60)

# Contagem absoluta no teste
contagem_teste = y_test.value_counts().sort_index()
print("\nContagem absoluta:")
print(f"  Nao viral (0): {contagem_teste[0]} registros")
print(f"  Viral (1): {contagem_teste[1]} registros")

# Proporcao percentual no teste
proporcao_teste = y_test.value_counts(normalize=True).sort_index() * 100
print("\nProporcao percentual:")
print(f"  Nao viral (0): {proporcao_teste[0]:.2f}%")
print(f"  Viral (1): {proporcao_teste[1]:.2f}%")

# ============================================
# 8. VERIFICACAO DA ESTRATIFICACAO
# ============================================
print("\n" + "="*60)
print("VERIFICACAO DA ESTRATIFICACAO")
print("="*60)

print("\nComparacao das proporcoes:")
print(f"\n  Classe 0 (Nao viral):")
print(f"    Dataset completo: {proporcao_total[0]:.2f}%")
print(f"    Treino: {proporcao_treino[0]:.2f}%")
print(f"    Teste: {proporcao_teste[0]:.2f}%")

print(f"\n  Classe 1 (Viral):")
print(f"    Dataset completo: {proporcao_total[1]:.2f}%")
print(f"    Treino: {proporcao_treino[1]:.2f}%")
print(f"    Teste: {proporcao_teste[1]:.2f}%")

print("\n" + "="*60)
print("PREPARACAO CONCLUIDA COM SUCESSO!")
print("="*60)

# ============================================
# OBSERVACOES
# ============================================
print("\nProximos passos:")
print("  1. Aplicar TF-IDF nas features de texto (X_train e X_test)")
print("  2. Treinar modelo de classificacao")
print("  3. Avaliar performance no conjunto de teste")
print("\n" + "="*60)
