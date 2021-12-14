### API para Clusterização

Api que aplica Kmeans à uma série de frases. Para isso usei um [embedding](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3) treinado.

### Clustering
Recebe como entrada:

rota: POST /clustering/?n_clusters=<i>total_de_clusters</i>
```json
{
    "messages":[
        "Isso é um teste",
        "Isso é outro teste",
        "Um terceiro teste pra variar",
        "Eu odeio biscoito",
        "Isso não tem nada haver",
        "Colher de pau, assubanipal",
        "Eu gosto de comer biscoito",
        "Eu quero ir pra praia",
        "Eu quero comer",
        "Maíra é minha esposa",
        "De abóbora com melão faz melancia"
    ]
}
```
Recebe como resposta um uid da requisição:
```json
{
    "id": "e05d09a2-fb9c-48fe-be8f-5b878a8c539e"
}
```
Para ter o resultado da clusterização, basta checar. Uma vez que você resgate o resultado ele não estará mais disponível no servidor, então é bom que guarde.
Também para computar os clusters pode demorar um pouco, então persista fazendo vários requests até conseguir resgatar seu resultado.

rota: GET /clustering/c2893574-7952-415d-a9af-b84698a726bf
```json
{
  "data": {
    "clusters": [
      {
        "phrase": "Cadê o carro que não chega?",
        "cluster": "0",
        "nearest_word": "jesus",
        "frequent_words": ["vem","jesus","cadê"]
      },
      ...,
      {
        "phrase": "A ineternet revolucionou o mundo",
        "cluster": "2",
        "nearest_word": "chocolate",
        "frequent_words": ["gosto","chocolate","quero"]
      }
    ],
    "cluster_centers": [
      [
        0.026613954454660416,
        ...
      ],
      ...
    ]
  }
}
```

### Knowledge
Essa rota, a partir do conhecimento prévio de uma intenção de referência, e de
uma análise feita pela rota clustering, infere qual dos clusters
pode ser intendido como uma intenção. A estratégia apresentada a seguir é ingênua, e deve ser aplicado somente
se for usado o mesmo tipo de embending para intenção de referência e clusters.

Isso é feito calculando o ponto médio do conjunto de frases que representam
a intenção de referência. A partir desse ponto médio, é obtido a distância
média entre ele e cada frase da intenção. 

O mesmo esquema é aplicado a cada cluster encontrado pela análise. Se algum cluster
apresentar a distância média menor que a intenção, consideramos ele uma possível intenção

rota: POST /knowledge/
```json
{
    "data": {
        "clusters": [
          {
            "phrase": "Cadê o carro que não chega?",
            "cluster": "0",
            "nearest_word": "jesus",
            "frequent_words": ["vem","jesus","cadê"]
          },
          ...,
          {
            "phrase": "A ineternet revolucionou o mundo",
            "cluster": "2",
            "nearest_word": "chocolate",
            "frequent_words": ["gosto","chocolate","quero"]
          }
        ],
        "cluster_centers": [
          [
            0.026613954454660416,
            ...
          ],
          ...
        ]
    },
    "intent_reference": [
        "horário",
        ...,
        "qual o horario de funcionamento"
    ]
}
```

Esse post tem como resposta o seguinte json:

````json
{
    "data": [
        {
            "cluster": 0,
            "threshold": 0.7663090999764368,
            "intent_test": "IGNORED"
        },
        {
            "cluster": 1,
            "threshold": 0.30518764018629285,
            "intent_test": "CANDIDATE"
        },
        {
            "cluster": 2,
            "threshold": 0.7969371948278763,
            "intent_test": "IGNORED"
        }
    ]
}
````
Onde "CANDIDATE" é todo cluster que tem uma distância média menor do que a
intenção de referência.