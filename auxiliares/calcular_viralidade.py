import pandas as pd
import os

# Configuração de paths
caminho_dataset = r"\\wsl.localhost\Ubuntu\home\filipehcunha\VIRALITY_BCC325\resultados\dataset_taxa_engajamento.csv"
caminho_temp = r"\\wsl.localhost\Ubuntu\home\filipehcunha\VIRALITY_BCC325\resultados\temp_dataset_com_viralidade.csv"

print("=== Calculando Viralidade dos Vídeos ===\n")

# 1. Carregar o dataset
print(f"Carregando dataset de: {caminho_dataset}")
df = pd.read_csv(caminho_dataset)
print(f"Total de vídeos carregados: {len(df)}\n")

# 2. Verificar colunas necessárias
colunas_necessarias = ['Curtidas', 'Comentários', 'Compartilhamentos', 'Alcance']
for coluna in colunas_necessarias:
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no dataset!")

# 3. Calcular a taxa de engajamento
print("Calculando taxa de engajamento...")
# Evitar divisão por zero substituindo 0 por NaN no alcance
df['taxa_engajamento'] = (
    (df['Curtidas'] + df['Comentários'] + df['Compartilhamentos']) / 
    df['Alcance'].replace(0, pd.NA)
)

# Preencher valores NaN (onde alcance era 0) com 0 na taxa de engajamento
df['taxa_engajamento'] = df['taxa_engajamento'].fillna(0)

print(f"Taxa de engajamento calculada para {len(df)} vídeos\n")

# 4. Ordenar por taxa de engajamento (maior para menor)
print("Ordenando vídeos por taxa de engajamento...")
df_ordenado = df.sort_values(by='taxa_engajamento', ascending=False).reset_index(drop=True)

# 5. Definir o threshold do top 20%
total_videos = len(df_ordenado)
threshold_index = int(total_videos * 0.20)  # Top 20%

print(f"Total de vídeos: {total_videos}")
print(f"Top 20% corresponde aos primeiros {threshold_index} vídeos\n")

# 6. Criar a coluna 'viral'
# Top 20% = 1 (viral), Restante 80% = 0 (não viral)
df_ordenado['viral'] = 0
df_ordenado.loc[:threshold_index-1, 'viral'] = 1

print("Coluna 'viral' criada:")
print(f"  - Vídeos virais (1): {df_ordenado['viral'].sum()}")
print(f"  - Vídeos não virais (0): {(df_ordenado['viral'] == 0).sum()}\n")

# 7. Estatísticas
print("=== Estatísticas da Taxa de Engajamento ===")
print(f"Média: {df_ordenado['taxa_engajamento'].mean():.4f}")
print(f"Mediana: {df_ordenado['taxa_engajamento'].median():.4f}")
print(f"Desvio padrão: {df_ordenado['taxa_engajamento'].std():.4f}")
print(f"Mínimo: {df_ordenado['taxa_engajamento'].min():.4f}")
print(f"Máximo: {df_ordenado['taxa_engajamento'].max():.4f}\n")

if threshold_index > 0:
    taxa_min_viral = df_ordenado.loc[threshold_index-1, 'taxa_engajamento']
    print(f"Taxa de engajamento mínima para ser viral: {taxa_min_viral:.4f}\n")

# 8. Salvar temporariamente
print(f"Salvando resultado temporário em: {caminho_temp}")
df_ordenado.to_csv(caminho_temp, index=False, encoding='utf-8-sig')
print("Arquivo temporário salvo com sucesso!\n")

# 9. Verificar se tudo está correto antes de sobrescrever
print("Verificando integridade dos dados...")
df_verificacao = pd.read_csv(caminho_temp)

# Verificações básicas
assert len(df_verificacao) == len(df), "Número de linhas diferente!"
assert 'taxa_engajamento' in df_verificacao.columns, "Coluna 'taxa_engajamento' não encontrada!"
assert 'viral' in df_verificacao.columns, "Coluna 'viral' não encontrada!"
assert df_verificacao['viral'].isin([0, 1]).all(), "Coluna 'viral' contém valores inválidos!"

print("Verificação concluída. Dados estão corretos!\n")

# 10. Sobrescrever o arquivo original
print(f"Salvando resultado final em: {caminho_dataset}")
df_ordenado.to_csv(caminho_dataset, index=False, encoding='utf-8-sig')

# 11. Remover arquivo temporário
if os.path.exists(caminho_temp):
    os.remove(caminho_temp)
    print("Arquivo temporário removido.\n")

print("=== Processo Concluído com Sucesso! ===")
print("\nColunas adicionadas ao dataset:")
print("  1. 'taxa_engajamento': Taxa calculada pela fórmula (curtidas + comentários + compartilhamentos) / alcance")
print("  2. 'viral': 1 para top 20% (viral), 0 para restante 80% (não viral)")
print(f"\nDataset final salvo em: {caminho_dataset}")
print("\nPrimeiros 10 vídeos virais:")
print(df_ordenado[df_ordenado['viral'] == 1][['Identificação do post', 'taxa_engajamento', 'viral']].head(10))
