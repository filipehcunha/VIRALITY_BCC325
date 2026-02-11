# 📊 Modelagem Preditiva de Viralidade em Vídeos Curtos

Este repositório contém a implementação do projeto desenvolvido na disciplina BCC325 (Inteligência Artificial) pelos alunos Filipe Hermenegildo e Julia Gonzaga.

## Requisitos

- Python 3.8+
- pandas
- scikit-learn
- joblib

## Estrutura do Repositório
```bash
├── main.py                  # Script principal de treinamento e avaliação
├── dataset.csv              # Dataset no formato CSV
├── modelo_viralidade.pkl    # Modelo treinado (gerado após execução)
├── vetor_tfidf.pkl          # Vetorizador TF-IDF salvo (gerado após execução)
├── README.md                # Documentação do projeto
```

## Instalação do Ambiente

### Instalação do Ambiente: Criar ambiente virtual (opcional)
python -m venv venv
```bash
python -m venv venv
```

### Ativar o ambiente Windows

### Instalar dependências
```bash
pip install pandas scikit-learn joblib
```

### Como Executar

```bash
python main.py
```

## Reutilizando o Modelo Treinado
```bash
import joblib

modelo = joblib.load('modelo_viralidade.pkl')
tfidf_vectorizer = joblib.load('vetor_tfidf.pkl')
```

