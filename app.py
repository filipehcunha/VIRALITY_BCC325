"""
APLICAÇÃO STREAMLIT - PREDIÇÃO DE VIRALIDADE DE VÍDEOS
========================================================

Esta aplicação permite ao usuário:
- Fazer upload de um vídeo
- Gerar automaticamente a descrição visual do vídeo
- Prever se o vídeo tem potencial viral usando modelo de ML pré-treinado
- Visualizar a probabilidade de viralização
"""

import streamlit as st
import joblib
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Importar a função de análise visual do módulo auxiliar
from auxiliares.analise_visual import analyze_video_frame


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Predição de Viralidade",
    page_icon="🎬",
    layout="centered"
)


# ============================================================================
# FUNÇÃO PARA CARREGAR MODELO E VETORIZADOR (COM CACHE)
# ============================================================================
@st.cache_resource
def carregar_modelo_e_vetorizador():
    """
    Carrega o modelo treinado e o vetorizador TF-IDF.
    Usa @st.cache_resource para carregar apenas uma vez.
    """
    try:
        # Carregar modelo de classificação
        modelo = joblib.load('modelo_viralidade.pkl')
        
        # Carregar vetorizador TF-IDF
        tfidf_vectorizer = joblib.load('vetor_tfidf.pkl')
        
        return modelo, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Erro ao carregar arquivos: {e}")
        st.error("Certifique-se de que os arquivos modelo_viralidade.pkl e vetor_tfidf.pkl estão no diretório.")
        st.stop()


# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================
def main():
    # Título da aplicação
    st.title("🎬 Predição de Viralidade de Vídeos")
    st.markdown("---")
    
    st.markdown("""
    Esta aplicação analisa vídeos e prevê se eles têm potencial **viral** 
    usando Inteligência Artificial.
    
    **Como funciona:**
    1. Faça upload de um vídeo
    2. A IA analisa o conteúdo visual automaticamente
    3. O modelo prevê a probabilidade de viralização
    """)
    
    st.markdown("---")
    
    # Carregar modelo e vetorizador
    with st.spinner("🔄 Carregando modelo de IA..."):
        modelo, tfidf_vectorizer = carregar_modelo_e_vetorizador()
    
    st.success("✅ Modelo carregado com sucesso!")
    
    # ========================================================================
    # UPLOAD DE VÍDEO
    # ========================================================================
    st.subheader("📤 Faça upload do seu vídeo")
    
    video_file = st.file_uploader(
        "Selecione um arquivo de vídeo",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Formatos aceitos: MP4, AVI, MOV, MKV"
    )
    
    if video_file is not None:
        # Exibir preview do vídeo
        st.video(video_file)
        
        # Botão para processar
        if st.button("🔮 Analisar Viralidade", type="primary"):
            
            # ================================================================
            # SALVAR VÍDEO TEMPORARIAMENTE
            # ================================================================
            with st.spinner("💾 Salvando vídeo temporariamente..."):
                # Criar arquivo temporário
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(video_file.name).suffix
                ) as tmp_file:
                    tmp_file.write(video_file.read())
                    video_path = tmp_file.name
            
            try:
                # ============================================================
                # GERAR DESCRIÇÃO VISUAL
                # ============================================================
                st.markdown("---")
                st.subheader("🔍 Análise Visual")
                
                with st.spinner("🤖 Analisando conteúdo visual do vídeo..."):
                    try:
                        # Usar a função do módulo auxiliar
                        descricao_visual = analyze_video_frame(video_path)
                        
                        # Exibir descrição gerada
                        st.success("✅ Descrição visual gerada!")
                        st.info(f"**Descrição:** {descricao_visual}")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao analisar vídeo: {e}")
                        st.stop()
                
                # ============================================================
                # VETORIZAR DESCRIÇÃO COM TF-IDF
                # ============================================================
                with st.spinner("🔢 Vetorizando descrição..."):
                    # IMPORTANTE: Usar apenas transform (NÃO fit_transform)
                    # O vetorizador já foi treinado no conjunto de treino
                    descricao_tfidf = tfidf_vectorizer.transform([descricao_visual])
                
                # ============================================================
                # FAZER PREDIÇÃO
                # ============================================================
                with st.spinner("🎯 Realizando predição..."):
                    # Prever classe (0 = não viral, 1 = viral)
                    predicao = modelo.predict(descricao_tfidf)[0]
                    
                    # Obter probabilidades
                    probabilidades = modelo.predict_proba(descricao_tfidf)[0]
                    prob_nao_viral = probabilidades[0]
                    prob_viral = probabilidades[1]
                
                # ============================================================
                # EXIBIR RESULTADOS
                # ============================================================
                st.markdown("---")
                st.subheader("📊 Resultado da Predição")
                
                # Determinar classificação
                if predicao == 1:
                    st.success("### 🔥 VÍDEO COM POTENCIAL VIRAL!")
                    st.balloons()
                else:
                    st.warning("### 📉 Vídeo sem potencial viral")
                
                # Exibir probabilidades
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Probabilidade de Viralizar",
                        value=f"{prob_viral * 100:.2f}%",
                        delta=f"{(prob_viral - prob_nao_viral) * 100:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Probabilidade de NÃO Viralizar",
                        value=f"{prob_nao_viral * 100:.2f}%"
                    )
                
                # Barra de progresso visual
                st.progress(prob_viral)
                
                # ============================================================
                # INTERPRETAÇÃO
                # ============================================================
                st.markdown("---")
                st.subheader("💡 Interpretação")
                
                if prob_viral >= 0.75:
                    st.success("🎉 **Alta probabilidade** de viralização! Este vídeo tem grande potencial.")
                elif prob_viral >= 0.50:
                    st.info("✨ **Probabilidade moderada** de viralização. O vídeo pode ter sucesso.")
                else:
                    st.warning("💭 **Baixa probabilidade** de viralização. Considere ajustar o conteúdo.")
                
            finally:
                # ============================================================
                # LIMPAR ARQUIVO TEMPORÁRIO
                # ============================================================
                if os.path.exists(video_path):
                    os.unlink(video_path)
    
    else:
        st.info("👆 Faça upload de um vídeo para começar a análise")
    
    # ========================================================================
    # RODAPÉ
    # ========================================================================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <small>
                Desenvolvido com ❤️ usando Streamlit | 
                Modelo: Regressão Logística + TF-IDF
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# EXECUTAR APLICAÇÃO
# ============================================================================
if __name__ == "__main__":
    main()
