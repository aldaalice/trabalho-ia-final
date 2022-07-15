# Trabalho-final-de-IA1

## Descrição:
  O presente trabalho visa demonstrar a implementação de três diferentes  métodos de machine learning para o reconhecimento e separação de três  diferentes tipos de  plantas, usando-se para isso as informações sobre o comprimento e a largura das sépalas de cada espécie, bem como os dados sobre a largura e o comprimento de das pétalas delas. 
  Como fonte de dados nós utilzamos o dataset Iris, sendo este muito conhecido e utilizado pela comunidade acadêmica para o estudo de diferentes tecnicas de análise de dados e de machine learning.
  Para o primeiro modelo nós utilizamos o método Árvore de Decisões. Optamos por começar por este modelo, pois durante os estudos e pesquisas vimos que ele é um modelo relativamente simples, de fácil entendimento e robusto. 
  Após a implementação e análise dos resultados passamos para a implemntação do modelo KNN, este modelo por sua vez possui uma um funcionamento mais complexo que o de Árvore de Decisões, uma vez que o mesmo faz uso da similaridade recurso e isso em alguns casos, onde os dados pussuem valores muito próximos uns dos outros pode causar problemas para que o modelo consiga fazer a separação dos diferentes tipos de dados.
  E por último fizemos o uso do SVM que é muito utilizado para para fazer clacificações, esse método apesar de mais complexo é considerado um método muito robusto e versátil, uma vez que ele pode ser utizado com uma quantidade um pouco menor de dados, até uma quantidade muito grande de dados.
  
 ## Ambiente de desenvolvimento:
   Para treinar e implementar os modelos nós escolhemos fazer uso da plataforma Coloaboratory (https://colab.research.google.com/?utm_source=scs-index), isso porque achamos mais interessante desenvolver o trabalho de equipe nela. 
   
   ## Bibliotecas instaladas:
   Para instalar as bibliotecas necessárias fizemos o uso do instalador (!pip install 'nome da biblioteca desejada'). 
   Já as bibliotecas utizadas foram: pandas, numpy, matplotlib, seaborn e sklearn.
   Por fim, para facilitar a execução dos modelos geramos um arquivo 'txt' com o comando (!pip freeze > requirements.txt) no Colaboratory, dessa forma foi gerado um arquivo que contém todas as bibliotecas utiladas, bem como susas respectivas verções.
