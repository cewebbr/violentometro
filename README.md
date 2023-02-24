# Violentometer - a study on the applicability of AI to the measurement of agression levels on the Web

This a broad-scope research project that revolves around the investigation of possibilities and challenges
faced by the application of Artificial Intelligence on the monitoring of online electoral violence directed
to candidates in the Brazilian 2022 elections, carried over with data collected from Twitter.

### Project outline

This research project is composed of the folowing smaller sub-projects:

1. An analysis of the Brazilian Superior Electoral Court (TSE) data on the use of web platforms by
   electoral candidates, and of the relationships between this data and the candidates' profiles.
2. The training of Machine Learning (ML) and Artificial Intelligence (AI) models for the identification
   of violence (i.e. hate speech and offensive comments) on texts.
3. The construction of a 24/7 system for capturing mentions on Twitter to a specified list of users and for
   estimating in real time the amount of aggressions directed to them.
4. The annotation by specialists of a sample of captured tweets in terms of violence and other relevant information.
5. An investigation of the model's performance and biases with respect to different social groups, violence
   types, targets of the tweets and different sets of annotators.
6. The development of a tool for estimating and displaying on the Web the evolution of violence levels directed
   to female candidates during the Brazilian 2022 election campaign period. 

The AI developed in this project was based on the one from
[cewebbr/Bert_HateSpeech_Classification](https://github.com/cewebbr/Bert_HateSpeech_Classification).


## Content and structure of the project

In this project you will find:


* an experimental AI model for identification of violent texts;
* a ML model for identifying the target of a comment on Twitter (if the correspondent or a third party);
* about 1,100 newly annotated tweets in terms of violence and other characteristics;
* notebooks that go through the process of training and testing the models above;
* notebooks that download public data about the Brazilian election and analyses them;
* a code for an automatic tweet capturing system;
* tutorials that go through specific parts of this project.

### Estructure
    .
    ├── analises            <- Jupyter notebooks containing data analysis
    |   ├── modelos         <- Jupyter notebooks for training the AI and ML models
    |   └── 00_indice.ipynb <- Index of all analysis and model building notebooks
    ├── dados               <- Data used in the project
    ├── docs                <- Documents (PDFs)
    ├── modelos             <- Actual AI and ML models saved to joblib or h5 files
    ├── src                 <- Python scripts and modules build for and used by the project
    ├── scripts             <- Shell scripts for deploying monitoring systems
    ├── tutoriais           <- Tutorials for reproducing specific parts of the project
    ├── tweets              <- Tweet capture config and storage place for the tweets
    ├── webpage             <- Data produced by the system to be used by a webpage 
    ├── LICENSE             <- License for this project
    ├── README.md           <- This document
    └── requirements.txt    <- Required python packages


## Perfis de teste

Para testar os métodos e modelos de detecção de violência, coletamos exemplos de texto da plataforma Twitter que
mencionavam alguns perfis de teste: os das candidaturas a deputados federal e estadual nas eleições de 2022.
Esses perfis foram obtidos dos [dados abertos do TSE](https://dadosabertos.tse.jus.br/dataset/), informados pelas
candidaturas durante seu registro, de forma voluntária. Os perfis do Twitter informados ao TSE foram complementados:

1. Por uma lista dos perfis de deputados federais concorrendo à reeleição;

2. Identificando a existência de perfis no Twitter com nome igual a perfis de candidaturas no Instagram, listados nos dados do TSE.
   Os perfis identificados foram filtrados por um modelo de machine learning para remover homônimos (veja o arquivo
   [analises/identificando_contas_twitter_pelo_instagram.ipynb](analises/identificando_contas_twitter_pelo_instagram.ipynb)).

A lista de perfis de candidaturas foi complementada quando a captura já estava em funcionamento. Os períodos nos quais
cada versão da lista foi utilizada estão registrados no arquivo
[tweets/logs/twitter_id_pool_sources.log](tweets/logs/twitter_id_pool_sources.log). Todas as versões utilizadas
encontram-se na pasta [dados/processados](dados/processados).


## Sistema de monitoramento e captura de tweets

O sistema de monitoramento e captura de tweets, escrito em Python, está disponível no arquivo
[src/tweet_capture.py](src/tweet_capture.py). Ele utiliza o endpoint
[mentions](https://developer.twitter.com/en/docs/twitter-api/tweets/timelines/api-reference/get-users-id-mentions)
da [API do Twitter](https://developer.twitter.com/en/docs/twitter-api)
para capturar, a cada 3 horas, os tweets produzidos nas últimas 3 horas em resposta ou que citam as candidaturas. É importante notar
que o endpoint [deixa de retornar alguns tweets](https://twittercommunity.com/t/missing-mentioned-tweets-from-the-user-timeline-from-api-get-2-users-id-mentions/169849)
que, a princípio, deveria. Além disso, a resposta do endpoint se limita aos 800 tweets mais recentes, o que torna a captura incompleta
para os casos de mais de 800 menções feitas num período de 3 horas.

Optamos por realizar a captura de maneira amostral. A cada 3 horas, sorteamos um terço dos perfis de candidaturas e realizamos a captura
conforme descrito acima. Para evitar duplicidade nos tweets capturados, os perfis utilizados na captura anterior são ignorados ao
sortear os perfis da captura seguinte.

## IA de identificação de discursos violentos

### O método

A metodologia usada para identificar discursos violentos foi baseada em <https://github.com/diogocortiz/NLP_HateSpeech_Classifier>.
As principais mudanças aplicadas foram:

* Pacote Tensorflow no lugar do PyTorch;
* Modelo BERTimbau no lugar do BERT multilingual;
* Dados de Pelle & Moreira (2017) adicionados aos dados de Fortuna et al. (2019) para treinamento e validação.

#### O modelo ajustado

O modelo de detecção de discurso de ódio disponibilizado aqui é um modelo transformer de
arquitetura BERT pré-treinado para tarefas em português: [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
Ele foi ajustado para classificação binária de textos entre aqueles que contém e que não
contém discurso de ódio a partir das bases disponibilizadas por
[Fortuna et. al (2019)](https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset) e
[Pelle & Moreira (2017)](https://github.com/rogersdepelle/OffComBR). 


#### Dados utilizados

Ambos os trabalhos acima, que fornecem exemplos para treinamento de detecção de discurso de ódio,
utilizam três anotadores para classificar cada exemplo.
Para os exemplos originados de [Pelle & Moreira (2017)](https://github.com/rogersdepelle/OffComBR),
que possuem uma concordância entre os anotadores de 0.71 ([Fleiss' Kappa](https://en.wikipedia.org/wiki/Fleiss'_kappa)),
escolhemos como discurso de ódio os exemplos com duas ou mais classificações nesse sentido, sendo que os
demais foram definidos como sem discurso de ódio.

No caso de [Fortuna et. al (2019)](https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset), cuja
concordância ([Fleiss' Kappa](https://en.wikipedia.org/wiki/Fleiss'_kappa)) foi de 0.17, escolhemos como
exemplos de discurso de ódio as instâncias com três classificações nesse sentido; e os com, no máximo,
uma classificação do tipo foram considerados livres desse tipo de discurso. A amostra resultado dessas
seleções -- e utilizada para treinar o modelo disponibilizado aqui -- encontra-se em
[dados/processados/hatespeech_fortuna3+offcombr2.csv](dados/processados/hatespeech_fortuna3+offcombr2.csv).

### Usando o modelo via módulo

Para facilitar o uso do modelo para previsão (i.e. classificação de textos entre discurso de ódio ou não),
criamos o módulo [src/speechwrapper.py](src/speechwrapper.py).

#### Instalação

O módulo utiliza os pacotes de Python: numpy, pandas, tensorflow, transformers e datasets
(os dois últimos do [Hugging Face](https://huggingface.co)). Para avaliar o modelo no notebook
[analises/modelo_bert_com_tensorflow.ipynb](analises/modelo_bert_com_tensorflow.ipynb), utilizamos
métricas do scikit-learn. Para instalar esses pacotes, você pode usar o comando:

    pip install -r requirements.txt

#### Uso do módulo

Dentro da pasta `src`, começe importando o módulo:

    import speechwrapper as sw

Em seguida, carregue o modelo via wrapper, indicando a pasta que contém o modelo (arquivo em formato `.h5`):

    model = sw.HateSpeechModel('../modelos/bertimbau-hatespeech-v01')

Nesse momento, o modelo também irá baixar o tokenizador do BERTimbau, caso ele não esteja presente na sua
máquina.

O objeto `model` possui os métodos `.predit_proba()` e `.predict_class()`, que recebem de input uma string
com o texto ou uma lista de textos (strings) a serem avaliados. O primeiro retorna a probabilidade de cada
texto ser discurso de ódio, e o segundo retorna, para cada texto e de acordo com o limiar (_threshold_)
estabelecido: 1, se o modelo considerá-lo discurso de ódio; e 0, caso contrário.

    exemplos = ['To te esperando com um cafezinho, pode chegar',
                'Não gostei nada da sua última proposta, com exceção, talvez, da parte 2.']

    model.predict_proba(exemplos)
    # Output: array([0.09299637, 0.00845698], dtype=float32) 


## Identificação do alvo (ou objeto) do tweet

Após o início do projeto, percebemos ser comum que um candidato mencione um adversário em um tweet e que
pessoas respondam ao candidato com ataques direcionados ao adversário, e não ao candidato em si.
Nesses casos, a IA descrita acima identifica os ataques como violência, mas não os diferencia em termos do alvo. Para fazer essa
diferenciação, criamos um modelo [Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
(veja o notebook [analises/modelo_baseline_objeto_do_tweet.ipynb](analises/modelo_baseline_objeto_do_tweet.ipynb)).
Duas versões desse modelo, sendo a segunda treinada em mais dados, estão disponíveis na pasta
[modelos/](modelos/).


## Finalmentes

### Aviso de privacidade e proteção de dados

Em conformidade à Lei Geral de Proteção de Dados Pessoais ([LGPD](http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)),
não disponibilizamos aqui os tweets coletados. Estes podem ser disponibilizados para fins de pesquisa mediante assinatura
de termo de responsabilidade. Este projeto segue o aviso de privacidade disponível em
[docs/juridico/Aviso_privacidade_Violentometro_Ceweb.pdf](docs/juridico/Aviso_privacidade_Violentometro_Ceweb.pdf).

### Contato

Para mais informações, entrar em contato com [Henrique S. Xavier](http://henriquexavier.net) (<https://github.com/hsxavier>).


