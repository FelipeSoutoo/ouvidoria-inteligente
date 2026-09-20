import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Ouvidoria Inteligente",
    page_icon="📢",
    layout="wide"
)

st.title("📢 Ouvidoria Inteligente")
st.write(
    "Triagem semântica de manifestações cidadãs utilizando "
    "embeddings e similaridade de cosseno."
)


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data
def carregar_dados():
    caminho = Path(__file__).parent / "manifestacoes.json"

    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    return pd.DataFrame(dados)


df = carregar_dados()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configurações")

modelo_nome = st.sidebar.selectbox(
    "Modelo de embeddings",
    [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2"
    ]
)

top_k = st.sidebar.slider(
    "Quantidade de resultados (Top-K)",
    min_value=1,
    max_value=10,
    value=5
)


# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================

@st.cache_resource
def carregar_modelo(nome_modelo):
    return SentenceTransformer(nome_modelo)


modelo = carregar_modelo(modelo_nome)


# ============================================================
# EMBEDDINGS DA BASE
# ============================================================

@st.cache_data
def gerar_embeddings(textos, nome_modelo):
    modelo_embeddings = carregar_modelo(nome_modelo)

    return modelo_embeddings.encode(
        textos,
        convert_to_numpy=True
    )


textos = df["texto"].tolist()

embeddings = gerar_embeddings(
    textos,
    modelo_nome
)


# ============================================================
# ABAS
# ============================================================

aba_busca, aba_base, aba_vetores, aba_chunking = st.tabs(
    [
        "🔎 Busca Semântica",
        "📋 Base de Manifestações",
        "📊 Espaço Vetorial",
        "✂️ Chunking"
    ]
)


# ============================================================
# ABA 1 — BUSCA SEMÂNTICA
# ============================================================

with aba_busca:

    st.header("Busca Semântica")

    consulta = st.text_input(
        "Digite uma manifestação ou descrição do problema:"
    )

    if consulta:

        embedding_consulta = modelo.encode(
            [consulta],
            convert_to_numpy=True
        )

        similaridades = cosine_similarity(
            embedding_consulta,
            embeddings
        )[0]

        indices = np.argsort(similaridades)[::-1][:top_k]

        st.subheader(f"Top {top_k} resultados")

        for indice in indices:

            score = float(similaridades[indice])

            if score > 0.7:
                cor = "green"
            elif score > 0.5:
                cor = "orange"
            else:
                cor = "red"

            st.markdown(
                f"### {df.iloc[indice]['id']} "
                f"— :{cor}[{score:.3f}]"
            )

            st.write(
                f"**Categoria:** "
                f"{df.iloc[indice]['categoria_oficial']}"
            )

            st.write(df.iloc[indice]["texto"])

            st.divider()


# ============================================================
# ABA 2 — BASE DE MANIFESTAÇÕES
# ============================================================

with aba_base:

    st.header("Base de Manifestações")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Matriz de Similaridade")

    if st.button("Gerar matriz de similaridade"):

        matriz_similaridade = cosine_similarity(embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))

        imagem = ax.imshow(
            matriz_similaridade,
            cmap="viridis",
            vmin=0,
            vmax=1
        )

        fig.colorbar(
            imagem,
            ax=ax,
            label="Similaridade de Cosseno"
        )

        ax.set_xticks(range(len(df)))
        ax.set_yticks(range(len(df)))

        ax.set_xticklabels(
            df["id"],
            rotation=90,
            fontsize=6
        )

        ax.set_yticklabels(
            df["id"],
            fontsize=6
        )

        ax.set_title(
            "Matriz de Similaridade entre as Manifestações"
        )

        ax.set_xlabel("Manifestações")
        ax.set_ylabel("Manifestações")

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# ABA 3 — ESPAÇO VETORIAL
# ============================================================

with aba_vetores:

    st.header("Espaço Vetorial")

    metodo_reducao = st.radio(
        "Método de redução dimensional",
        ["PCA", "t-SNE"],
        horizontal=True
    )

    if metodo_reducao == "PCA":

        redutor = PCA(
            n_components=2
        )

        coordenadas = redutor.fit_transform(
            embeddings
        )

    else:

        perplexidade = min(
            10,
            len(df) - 1
        )

        redutor = TSNE(
            n_components=2,
            perplexity=perplexidade,
            random_state=42,
            init="pca",
            learning_rate="auto"
        )

        coordenadas = redutor.fit_transform(
            embeddings
        )

    df_visualizacao = df.copy()

    df_visualizacao["Componente 1"] = coordenadas[:, 0]
    df_visualizacao["Componente 2"] = coordenadas[:, 1]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    categorias = df_visualizacao[
        "categoria_oficial"
    ].unique()

    for categoria in categorias:

        dados_categoria = df_visualizacao[
            df_visualizacao["categoria_oficial"]
            == categoria
        ]

        ax.scatter(
            dados_categoria["Componente 1"],
            dados_categoria["Componente 2"],
            label=categoria
        )

    ax.set_title(
        f"Espaço Vetorial das Manifestações — {metodo_reducao}"
    )

    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")

    ax.legend(
        title="Categoria"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "Pontos próximos indicam manifestações com "
        "representações vetoriais semelhantes. A visualização "
        "permite observar se manifestações da mesma categoria "
        "tendem a formar agrupamentos."
    )


# ============================================================
# ABA 4 — CHUNKING
# ============================================================

with aba_chunking:

    st.header("Chunking de Manifestações")

    texto_chunking = st.text_area(
        "Digite ou cole uma manifestação longa:",
        height=200
    )

    col1, col2 = st.columns(2)

    with col1:

        chunk_size = st.slider(
            "Chunk size",
            min_value=100,
            max_value=500,
            value=200,
            step=50
        )

    with col2:

        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=100,
            value=40,
            step=10
        )

    if chunk_overlap >= chunk_size:

        st.error(
            "O chunk overlap deve ser menor que o chunk size."
        )

    elif texto_chunking:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = splitter.split_text(
            texto_chunking
        )

        st.subheader(
            f"Chunks gerados: {len(chunks)}"
        )

        for numero, chunk in enumerate(
            chunks,
            start=1
        ):

            with st.expander(
                f"Chunk {numero}"
            ):
                st.write(chunk)
                st.caption(
                    f"{len(chunk)} caracteres"
                )

        if len(chunks) >= 2:

            st.subheader(
                "Embeddings dos Chunks"
            )

            embeddings_chunks = modelo.encode(
                chunks,
                convert_to_numpy=True
            )

            st.write(
                "Formato dos embeddings:",
                embeddings_chunks.shape
            )

            pca_chunks = PCA(
                n_components=2
            )

            coordenadas_chunks = (
                pca_chunks.fit_transform(
                    embeddings_chunks
                )
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.scatter(
                coordenadas_chunks[:, 0],
                coordenadas_chunks[:, 1]
            )

            for i in range(len(chunks)):

                ax.annotate(
                    f"Chunk {i + 1}",
                    (
                        coordenadas_chunks[i, 0],
                        coordenadas_chunks[i, 1]
                    )
                )

            ax.set_title(
                "Distribuição dos Chunks — PCA"
            )

            ax.set_xlabel(
                "Componente Principal 1"
            )

            ax.set_ylabel(
                "Componente Principal 2"
            )

            ax.grid(
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        elif len(chunks) == 1:

            st.info(
                "O texto gerou apenas um chunk. "
                "Insira um texto maior ou reduza o chunk size "
                "para visualizar os embeddings em PCA."
            )