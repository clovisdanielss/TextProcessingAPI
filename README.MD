### API para Clusterização

Api que aplica KNN à uma série de frases. Para isso usei um [embedding](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3) treinado.

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
Para ter o resultado da clusterização, basta checar.

rota: GET/clustering/c2893574-7952-415d-a9af-b84698a726bf
```json
{
    "data": [
        {
            "phrase": "Eu odeio biscoito",
            "cluster": "0",
            "nearest_word": "quero",
            "nearest_words": "biscoito,comer,quero"
        },
        {
            "phrase": "Eu gosto de comer biscoito",
            "cluster": "0",
            "nearest_word": "quero",
            "nearest_words": "biscoito,comer,quero"
        },
        {
            "phrase": "Eu quero ir pra praia",
            "cluster": "0",
            "nearest_word": "quero",
            "nearest_words": "biscoito,comer,quero"
        },
        {
            "phrase": "Eu quero comer",
            "cluster": "0",
            "nearest_word": "quero",
            "nearest_words": "biscoito,comer,quero"
        },
        {
            "phrase": "Isso não tem nada haver",
            "cluster": "1",
            "nearest_word": "nada",
            "nearest_words": "nada,haver"
        },
        {
            "phrase": "Isso é um teste",
            "cluster": "2",
            "nearest_word": "teste",
            "nearest_words": "teste,outro,terceiro"
        },
        {
            "phrase": "Isso é outro teste",
            "cluster": "2",
            "nearest_word": "teste",
            "nearest_words": "teste,outro,terceiro"
        },
        {
            "phrase": "Um terceiro teste pra variar",
            "cluster": "2",
            "nearest_word": "teste",
            "nearest_words": "teste,outro,terceiro"
        },
        {
            "phrase": "Colher de pau, assubanipal",
            "cluster": "3",
            "nearest_word": "maíra",
            "nearest_words": "colher,pau,assubanipal"
        },
        {
            "phrase": "Maíra é minha esposa",
            "cluster": "3",
            "nearest_word": "maíra",
            "nearest_words": "colher,pau,assubanipal"
        },
        {
            "phrase": "De abóbora com melão faz melancia",
            "cluster": "3",
            "nearest_word": "maíra",
            "nearest_words": "colher,pau,assubanipal"
        }
    ]
}
```