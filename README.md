# Ouvidoria Inteligente

Aplicação de Processamento de Linguagem Natural (NLP) para análise e triagem semântica de manifestações cidadãs.

O projeto utiliza diferentes técnicas de representação textual, embeddings, similaridade de cosseno, detecção de duplicatas e chunking para analisar manifestações recebidas por uma ouvidoria municipal.

## Objetivo

O objetivo do projeto é explorar técnicas de NLP capazes de identificar relações semânticas entre manifestações, auxiliando tarefas como:

- busca semântica;
- identificação de manifestações semelhantes;
- detecção de possíveis duplicatas;
- análise do espaço vetorial;
- processamento de manifestações longas por meio de chunking.

## Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Sentence Transformers
- LangChain Text Splitters
- Jupyter Notebook

## Base de Dados

O arquivo `manifestacoes.json` contém 40 manifestações cidadãs anonimizadas, distribuídas entre as categorias:

- infraestrutura;
- saúde;
- segurança;
- educação;
- meio ambiente.

A base utilizada neste projeto é sintética e contém casos de duplicidade semântica e manifestações longas para permitir a avaliação das técnicas implementadas.

## Análises Desenvolvidas

### Comparação de Representações

O notebook `analise_comparativa.ipynb` compara três formas de representação textual:

- Bag-of-Words (BoW);
- TF-IDF;
- Sentence Embeddings.

A similaridade de cosseno é utilizada para comparar pares de manifestações e observar as diferenças entre abordagens baseadas em palavras e representações semânticas.

### Detecção de Duplicatas

O notebook `deteccao_duplicatas.ipynb` utiliza embeddings e similaridade de cosseno para identificar possíveis manifestações duplicadas.

A implementação utiliza inicialmente um limiar de similaridade de `0.85` e também analisa o comportamento de diferentes valores de limiar, considerando verdadeiros positivos, falsos positivos e falsos negativos.

### Chunking

O notebook `chunking_manifestacoes.ipynb` analisa as cinco manifestações mais longas da base utilizando `RecursiveCharacterTextSplitter`.

São comparadas duas configurações:

- `chunk_size = 200` e `chunk_overlap = 40`;
- `chunk_size = 350` e `chunk_overlap = 70`.

Os embeddings dos chunks também são projetados em duas dimensões utilizando PCA para analisar sua distribuição no espaço vetorial.

## Aplicação Streamlit

O arquivo `app_ouvidoria.py` disponibiliza uma interface interativa dividida em quatro áreas:

### Busca Semântica

Permite inserir uma consulta e recuperar as manifestações semanticamente mais próximas, ordenadas pela similaridade de cosseno.

### Base de Manifestações

Exibe as manifestações cadastradas e permite gerar a matriz completa de similaridade entre os textos.

### Espaço Vetorial

Permite visualizar os embeddings das manifestações em duas dimensões utilizando PCA ou t-SNE, com identificação das categorias oficiais.

### Chunking

Permite inserir uma manifestação longa, configurar `chunk_size` e `chunk_overlap`, visualizar os chunks gerados e analisar seus embeddings com PCA.

## Instalação

Recomenda-se utilizar um ambiente virtual Python.

No Windows:

```powershell
python -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

## Executando a Aplicação

Com o ambiente virtual ativado, execute na raiz do projeto:

```powershell
streamlit run app_ouvidoria.py
```

O Streamlit iniciará a aplicação e disponibilizará o endereço local para acesso pelo navegador.

## Executando os Notebooks

Os notebooks estão disponíveis na pasta `notebooks/`.

Antes de executá-los, certifique-se de que o ambiente virtual do projeto está selecionado como kernel do Jupyter.

A ordem sugerida é:

1. `analise_comparativa.ipynb`
2. `deteccao_duplicatas.ipynb`
3. `chunking_manifestacoes.ipynb`

## Observações

Os valores de similaridade dependem do modelo de embeddings selecionado. Um alto valor de similaridade indica maior proximidade entre as representações vetoriais, mas não garante, isoladamente, que duas manifestações sejam duplicatas.

A definição de limiares deve considerar as características da base e o equilíbrio desejado entre falsos positivos e falsos negativos.