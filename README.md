# Violentômetro - Violência eleitoral nas redes

Módulo em Python que calcula a probabilidade de textos conterem
discurso de ódio.

## Estrutura do projeto

    .
    ├── README.md               <- Este documento
    ├── requirements.txt        <- Pacotes de python necessários
    ├── analises                <- Notebook com o treinamento do modelo
    ├── src                     <- Módulo em Python p/ usar o modelo
    ├── dados                   <- Dados utilizados no projeto
    ├── modelos                 <- Arquivos dos modelos treinados
    └── LICENSE                 <- Licença de uso e cópia
    

## Descrição do projeto

A metodologia deste projeto foi baseada em <https://github.com/diogocortiz/NLP_HateSpeech_Classifier>.
As principais mudanças aplicadas foram:

* Pacote Tensorflow no lugar do PyTorch;
* Modelo BERTimbau no lugar do BERT multilingual;
* Dados de Pelle & Moreira (2017) adicionados aos dados de Fortuna et al. (2019) para treinamento e validação.

### O modelo ajustado

O modelo de detecção de discurso de ódio disponibilizado aqui é um modelo transformer de
arquitetura BERT pré-treinado para tarefas em português: [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
Ele foi ajustado para classificação binária de textos entre aqueles que contém e que não
contém discurso de ódio a partir das bases disponibilizadas por
[Fortuna et. al (2019)](https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset) e
[Pelle & Moreira (2017)](https://github.com/rogersdepelle/OffComBR). 


### Dados utilizados

Ambos os trabalhos acima, que fornecem exemplos para treinamento de detecção de discurso de ódio,
utilizam três anotadores para classificar cada exemplo.
Para os exemplos originados de [Pelle & Moreira (2017)](https://github.com/rogersdepelle/OffComBR),
que possui uma concordância entre os anotadores de 0.71 ([Fleiss' Kappa](https://en.wikipedia.org/wiki/Fleiss'_kappa))
escolhemos como discurso de ódio os exemplos com duas ou mais classificações nesse sentido, sendo que os
demais foram definidos como sem discurso de ódio.

No caso de [Fortuna et. al (2019)](https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset), cuja
concordância ([Fleiss' Kappa](https://en.wikipedia.org/wiki/Fleiss'_kappa)) foi de 0.17, escolhemos como
exemplos de discurso de ódio as instâncias com três classificações nesse sentido; e os com, no máximo,
uma classificação do tipo foram considerados livres desse tipo de discurso. A amostra resultado dessas
seleções -- e utilizada para treinar o modelo disponibilizado aqui -- encontra-se em
[dados/processados/hatespeech_fortuna3+offcombr2.csv](dados/processados/hatespeech_fortuna3+offcombr2.csv).

## Usando o modelo via módulo

Para facilitar o uso do modelo para previsão (i.e. classificação de textos entre discurso de ódio ou não),
criamos o módulo [src/speechwrapper.py](src/speechwrapper.py).

### Instalação

O módulo utiliza os pacotes de Python: numpy, pandas, tensorflow, transformers e datasets
(os dois últimos do [Hugging Face](https://huggingface.co)). Para avaliar o modelo no notebook
[analises/modelo_bert_com_tensorflow.ipynb](analises/modelo_bert_com_tensorflow.ipynb), utilizamos
métricas do scikit-learn. Para instalar esses pacotes, você pode usar o comando:

    pip install -r requirements.txt

### Uso do módulo

Dentro da pasta `src`, começe importando o módulo:

    import speechwrapper as sw

Em seguida, carregue o modelo via wrapper, indicando a pasta que contém o modelo (arquivo em formato `.h5`):

    model = sw.HateSpeechModel('../modelos/bertimbau-hatespeech-v01')

Nesse momento, o modelo também irá baixar o tokenizador do BERTimbau, caso ele não esteja presente na sua
máquina.

O objeto `model` possui os métodos `.predit_proba()` e `.predict_class()`, que recebem de input uma string
com o texto ou uma lista de textos (strings) a serem avaliados. O primeiro retorna a probabilidade de cada
texto ser discurso de ódio, e o segundo retorna, para cada texto e de acordo com o limiar (_threshold_)
estabelecido: 1, se o modelo considerá-lo discurso de ódio; e 0 caso contrário.

    exemplos = ['To te esperando com um cafezinho, pode chegar', 'Não gostei nada da sua última proposta, com exceção, talvez, da parte 2.']
    model.predict_proba(exemplos)
    # Output: array([0.09299637, 0.00845698], dtype=float32) 

## Contato

Para mais informações, entrar em contato com [Henrique S. Xavier](http://henriquexavier.net) (<https://github.com/hsxavier>).