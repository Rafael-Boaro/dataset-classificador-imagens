# Classificador de Imagens com Transfer Learning (VGG16)

Classificador de imagens em Python/TensorFlow que compara duas abordagens — uma CNN treinada do zero e transfer learning com VGG16 — e mostra, para uma imagem nova, a classe prevista com a região de interesse (ROI) marcada na imagem via Grad-CAM.

## Sumário

- [Sobre o projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como funciona o pipeline](#como-funciona-o-pipeline)
- [Dataset](#dataset)
- [Requisitos](#requisitos)
- [Como executar](#como-executar)
- [Configuração](#configuração)
- [Saída esperada](#saída-esperada)
- [Solução de problemas](#solução-de-problemas)
- [Licença](#licença)
- [Créditos](#créditos)

## Sobre o projeto

Este projeto treina e compara dois classificadores de imagem:

1. **Baseline do zero** — uma CNN pequena (Conv2D + MaxPooling) treinada do início, sem nenhum conhecimento prévio.
2. **Transfer learning** — a VGG16 pré-treinada na ImageNet é usada como extratora de características (congelada), e só um classificador pequeno é treinado em cima dessas features.

No final, o notebook aplica o modelo em uma imagem nova e mostra o resultado com a classe identificada e um retângulo marcando a região da imagem que mais influenciou a decisão (ROI via Grad-CAM).

## Funcionalidades

- Download automático do dataset [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) caso nenhum dataset local seja encontrado.
- Pipeline `tf.data` com cache em disco (evita estourar a memória RAM, inclusive no Google Colab).
- Extração de bottleneck features da VGG16 (treino do classificador muito mais rápido que retreinar a rede toda).
- `EarlyStopping` nos dois treinos, evitando épocas desnecessárias.
- Detecção automática de GPU/CPU, com ajuste de threads quando não há GPU disponível.
- Visualização final: imagem + classe prevista + confiança + caixa delimitadora do ROI (Grad-CAM).

## Estrutura do repositório

```
dataset-classificador-imagens/
├── elemento_A/              # dataset próprio de exemplo — imagens de cachorros (classe A)
├── elemento_B/              # dataset próprio de exemplo — imagens de gatos (classe B)
└── transfer_learning.ipynb  # notebook principal, pronto para rodar no Google Colab
```

## Como funciona o pipeline

1. **Preparação do dataset** — procura `101_ObjectCategories` ou `Data_set` localmente; se não encontrar nenhum dos dois, baixa e extrai o Caltech-101 automaticamente.
2. **Indexação** — lista os caminhos das imagens e separa em treino (70%), validação (15%) e teste (15%).
3. **Baseline do zero** — treina a CNN pequena e avalia no conjunto de teste.
4. **Transfer learning** — carrega a VGG16 sem o topo (`include_top=False, pooling='avg'`), extrai as features de cada imagem uma única vez e treina um classificador enxuto em cima delas.
5. **Comparação** — plota loss/acurácia de validação dos dois modelos lado a lado.
6. **Predição com ROI** — aplica o modelo de transfer learning em uma imagem de teste (ou uma imagem própria, via `path_custom`) e mostra a imagem com a classe prevista, a confiança e a caixa do ROI desenhada sobre ela.

## Dataset

Por padrão, o notebook baixa o **Caltech-101** (101 categorias, ~9.000 imagens) automaticamente na primeira execução.

Este repositório também inclui um dataset próprio de duas classes, pronto para uso:

| Pasta | Classe | Imagens |
|---|---|---|
| `elemento_A/` | Cachorros | 145 |
| `elemento_B/` | Gatos | 127 |

Para usar esse dataset próprio em vez do Caltech-101, organize as pastas assim antes de rodar o notebook (o script procura por uma pasta chamada `Data_set`):

```
Data_set/
├── elemento_A/
└── elemento_B/
```

Para usar suas próprias imagens, basta seguir a mesma estrutura: uma subpasta por classe dentro de `Data_set/`.

## Requisitos

O notebook foi feito para rodar direto no Google Colab, sem instalação. Para rodar localmente, os pacotes necessários são:

```
tensorflow
numpy
matplotlib
```

```bash
pip install tensorflow numpy matplotlib
```

## Como executar

### Opção 1 — Google Colab (recomendado)

1. Clique no badge **Open in Colab** no topo deste README (ou abra `transfer_learning.ipynb` diretamente pelo Colab).
2. Rode todas as células, em ordem (`Ambiente de execução > Executar tudo`).
3. Se quiser usar suas próprias imagens em vez do Caltech-101, envie a pasta `Data_set/` para o Colab (ou monte o Google Drive, já configurado no notebook) antes de rodar a célula de download.

### Opção 2 — Localmente

```bash
git clone https://github.com/Rafael-Boaro/dataset-classificador-imagens.git
cd dataset-classificador-imagens
pip install tensorflow numpy matplotlib
jupyter notebook transfer_learning.ipynb
```

## Configuração

Os principais parâmetros ficam no topo do notebook e podem ser ajustados diretamente:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `IMG_SIZE` | `(224, 224)` | Resolução de entrada (exigida pela VGG16) |
| `BATCH_SIZE` | `32` | Tamanho do lote no treino |
| `FEATURE_BATCH_SIZE` | `64` | Tamanho do lote na extração de features (sem backprop) |
| `TRAIN_BASELINE_FROM_SCRATCH` | `True` | Mude para `False` para pular o modelo do zero e economizar tempo |
| `SEED` | `42` | Semente para resultados reproduzíveis |

## Saída esperada

Ao final da execução, o notebook exibe:

- Um resumo (`summary()`) de cada modelo treinado.
- As métricas de teste (loss e acurácia) do baseline e do transfer learning.
- Um gráfico comparando a curva de validação dos dois modelos.
- Uma imagem com a classe prevista, a confiança e um retângulo verde marcando o ROI identificado, por exemplo:

  ```
  Classe prevista: elemento_A (confiança: 92.4%)
  ```

## Solução de problemas

**Sessão do Colab reiniciando por falta de memória**
O cache das imagens é feito em disco (pasta `.tf_cache/`, recriada a cada execução), não em RAM. Se ainda assim faltar memória, mude `TRAIN_BASELINE_FROM_SCRATCH` para `False` para pular o treino do modelo do zero.

**Erro `HTTP Error 403: Forbidden` ao baixar o dataset**
O servidor do CaltechDATA bloqueia requisições sem um `User-Agent` de navegador. O notebook já envia um `User-Agent` válido por padrão; se o erro persistir, tente rodar a célula novamente (pode ser instabilidade temporária do servidor).

**`FileNotFoundError: Dataset root not found`**
Nenhuma pasta `101_ObjectCategories` ou `Data_set` foi encontrada e o download automático falhou. Verifique a conexão com a internet ou baixe o Caltech-101 manualmente a partir do [link oficial](https://data.caltech.edu/records/mzrjq-6wc02).

## Licença

Este repositório ainda não possui um arquivo de licença. Se você pretende permitir reuso do código por terceiros, considere adicionar uma licença (por exemplo, [MIT](https://choosealicense.com/licenses/mit/)).

## Créditos

- [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) — Fei-Fei Li, Marco Andreetto, Marc'Aurelio Ranzato, Pietro Perona (Caltech).
- [VGG16](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, pré-treinada na ImageNet.
- [TensorFlow / Keras](https://www.tensorflow.org/)
