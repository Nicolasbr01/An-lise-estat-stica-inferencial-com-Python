#!/usr/bin/env python
# coding: utf-8

# # Caderno de prática para intervalos de confiança usando dados do NHANES
# Este caderno lhe dará a oportunidade de praticar trabalhar com intervalos de confiança usando os dados do NHANES.
# 
# Você pode inserir seu código nas células que dizem "insira seu código aqui", e você pode digitar respostas para as perguntas nas células que dizem "Digite Markdown e Latex".
# 
# Note que a maior parte do código que você irá escrever abaixo é muito similar ao código que aparece no caderno de estudo de caso. Você precisará editar o código daquele caderno de maneiras pequenas para adaptá-lo às instruções abaixo.
# 
# Para começar, iremos usar os mesmos imports de módulos e ler os dados da mesma forma como fizemos no estudo de caso:

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import scipy.stats

da = pd.read_csv("nhanes_2015_2016.csv")


# ## Questão 1
# Restrinja a amostra para mulheres entre 35 e 50 anos, então use a variável de estado civil DMDMARTL para dividir essa amostra em dois grupos - mulheres que estão atualmente casadas e mulheres que não estão atualmente casadas. Dentro de cada um desses grupos, calcule a proporção de mulheres que completaram o ensino superior e calcule intervalos de confiança de 95% para cada uma dessas proporções.

# In[2]:


# Primeiro sempre inspecione os dados
da.head()


# In[3]:


# Obtenha as mulheres entre 35 e 50 anos. O valor de RIAGENDR é igual a 2 se o sujeito for do sexo feminino
middle_age_women = da.query('RIDAGEYR >= 35 & RIDAGEYR <= 50 & RIAGENDR == 2')

#Verifique se nossos dados estão corretos! Esta etapa não é necessária, mas é uma boa prática :)
assert np.all(middle_age_women['RIAGENDR'] == 2)
assert np.all(middle_age_women['RIDAGEYR'] <= 50)
assert np.all(middle_age_women['RIDAGEYR'] >= 35)


# In[4]:


# Dividir o grupo em mulheres casadas e não casadas
married_middle_age_women = middle_age_women.query('DMDMARTL == 1')
non_married_middle_age_women = middle_age_women.query('DMDMARTL != 1')

# Mais verificações de dados!
assert np.all(married_middle_age_women['DMDMARTL'] == 1)
assert np.all(non_married_middle_age_women['DMDMARTL'] != 1)


# In[9]:


# Agora verificamos se nossos dados são válidos e estão particionados corretamente. Nós
# agora podemos calcular os intervalos de confiança de 90% para a proporção que
# concluiu a faculdade. Isto é codificado na variável DMDEDUC2 e é igual a
# 5 se eles concluíram a faculdade ou superior
married_and_completed_college = married_middle_age_women['DMDEDUC2'] == 5
p_hat_married = np.mean(married_and_completed_college)
married_sample_size = married_and_completed_college.size
"A proporção de mulheres casadas de meia idade (N={}) que concluíram a faculdade é: {:.2f}".format(
    married_sample_size, 
    p_hat_married
)


# In[10]:


# Podemos fazer o mesmo com mulheres não casadas
non_married_and_completed_college = non_married_middle_age_women['DMDEDUC2'] == 5
p_hat_non_married = np.mean(non_married_and_completed_college)
non_married_sample_size = non_married_and_completed_college.size
"A proporção de mulheres de meia idade não casadas (N={}) que concluíram a faculdade é: {:.2f}".format(non_married_sample_size, p_hat_non_married)


# In[11]:


# Agora podemos calcular os intervalos de confiança. Lembre-se, para um bilateral
# intervalo de confiança, precisamos de 5% em cada uma das caudas e 95% do PPF nos dará
# este valor!
z_multiplier = scipy.stats.norm.ppf(q = 0.95)
married_standard_error = np.sqrt(p_hat_married * (1 - p_hat_married) / married_sample_size)
ci_married_lower_bound = p_hat_married - z_multiplier * married_standard_error
ci_married_upper_bound = p_hat_married + z_multiplier * married_standard_error
"Um intervalo de confiança de 90% para a proporção de mulheres casadas que concluíram a faculdade é ({:.2f}, {:.2f})".format(
    ci_married_lower_bound, 
    ci_married_upper_bound
)


# In[12]:


# Agora podemos fazer o mesmo com mulheres não casadas
z_multiplier = scipy.stats.norm.ppf(q = 0.95)
non_married_standard_error = np.sqrt(p_hat_non_married * (1 - p_hat_non_married) / non_married_sample_size)
ci_non_married_lower_bound = p_hat_non_married - z_multiplier * non_married_standard_error
ci_non_married_upper_bound = p_hat_non_married + z_multiplier * non_married_standard_error
"Um intervalo de confiança de 90% para a proporção de mulheres solteiras que concluíram a faculdade é ({:.2f}, {:.2f})".format(ci_non_married_lower_bound, ci_non_married_upper_bound)


# __Q1a.__ Identifique qual dos dois intervalos de confiança é mais amplo e explique por que isso acontece.

# A largura do intervalo de confiança para a proporção populacional de mulheres casadas que completaram o ensino superior é de oito pontos percentuais, enquanto o intervalo de confiança para mulheres não casadas é de sete pontos percentuais. O intervalo de confiança para mulheres casadas é maior, apesar de ter um tamanho de amostra maior, porque a estimativa $\hat{p}{casadas}$ está mais próxima de 50% do que $\hat{p}{não-casadas}$, então o erro padrão de $\hat{p}_{casadas}$ é maior.

# __Q1b.__ Escreva uma ou duas frases resumindo essas descobertas para um público que não sabe o que é um intervalo de confiança (o objetivo aqui é relatar a substância do que você aprendeu sobre como o estado civil e o nível de escolaridade estão relacionados, e não ensinar a uma pessoa o que é um intervalo de confiança). intervalo de confiança é).

# Podemos ver que, em média, uma mulher de meia-idade casada tem maior probabilidade de ter concluído a faculdade do que uma mulher de meia-idade que não é casada, e os intervalos de confiança para estas duas estimativas não se sobrepõem.

# ## Questão 2
# 
# Construa um intervalo de confiança de 95% para a proporção de mulheres fumantes. Construa um intervalo de confiança de 95% para a proporção de fumantes do sexo masculino. Construa um intervalo de confiança de 95% para a **diferença** entre essas duas proporções de gênero.

# In[15]:


# Primeiro, vamos preparar nossos dados
females = da.query('RIAGENDR == 2')
males = da.query('RIAGENDR == 1')

# Encontrei a variável tabagismo aqui: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/SMQ_I.htm#SMQ020
is_female_smoker = females['SMQ020'] == 1
is_male_smoker = males['SMQ020'] == 1


# In[16]:


# Para reduzir a chance de erros, vou criar uma função simples
# para calcular intervalos de confiança. Esta função executará exatamente as mesmas etapas que
# acima, então nada deve ser assustador :)
def MakeAConfidenceInterval(data, confidence):
    p_hat = np.mean(data)
    n = data.size
    z_multiplier = scipy.stats.norm.ppf(q = 1 - (1 - confidence) / 2)
    standard_error = np.sqrt(p_hat * (1 - p_hat) / n)
    lower_bound = p_hat - z_multiplier * standard_error
    upper_bound = p_hat + z_multiplier * standard_error
    return {"phat": p_hat, "n":n, "lower_bound":lower_bound, "upper_bound":upper_bound}


# In[18]:


female_smoking_ci = MakeAConfidenceInterval(data = is_female_smoker, confidence = 0.95)
"O IC para mulheres fumantes (N={}) é ({:.2f}, {:.2f})".format(
    female_smoking_ci['n'],
    female_smoking_ci['lower_bound'],
    female_smoking_ci['upper_bound']
)


# In[19]:


male_smoking_ci = MakeAConfidenceInterval(data = is_male_smoker, confidence = 0.95)
"Um IC para fumantes do sexo masculino (N={}) é ({:.2f}, {:.2f})".format(
    male_smoking_ci['n'],
    male_smoking_ci['lower_bound'],
    male_smoking_ci['upper_bound']
)


# In[20]:


# Agora podemos construir o intervalo de confiança para a diferença entre as proporções.
# Esta fórmula é bem longa, então vamos dividi-la em partes

# Primeiro calcule o erro padrão da diferença
standard_error = np.sqrt(
    female_smoking_ci['phat'] * (1 - female_smoking_ci['phat']) / female_smoking_ci['n'] + 
    male_smoking_ci['phat'] * (1 - male_smoking_ci['phat']) / male_smoking_ci['n']
)

# Obtenha o multiplicador z para nosso intervalo de confiança de 95%
z_multiplier = scipy.stats.norm.ppf(q = 0.975)

# Calcule a estimativa pontual para as diferenças entre a proporção de fumantes masculinos e femininos
difference = male_smoking_ci['phat'] - female_smoking_ci['phat']

# Calcula o limite inferior e o limite superior do intervalo de confiança
lower_bound = difference - z_multiplier * standard_error
upper_bound = difference + z_multiplier * standard_error
"Um IC para a diferença entre a proporção de fumantes masculinos menos a proporção de fumantes femininos é ({:.2f}, {:.2f})".format(
  lower_bound,
  upper_bound
)


# __Q2a.__ Por que pode ser relevante relatar as proporções de gênero separadas **e** a diferença entre as proporções de gênero?

# Quando olhamos para o intervalo de confiança dos homens que fumam e para a proporção de mulheres que fumam, vemos como eles se comportam marginalmente. Ou seja, vemos como eles se comportam em seu próprio grupo. Mas, para compará-los diretamente, também temos de calcular o intervalo de confiança para a diferença. Isso garante que as duas amostras sejam comparadas de forma justa e nós, tanto quanto possível, controlamos as diferenças de tamanho da amostra.

# __Q2b.__ Como a **largura** do intervalo de confiança para a diferença entre as proporções de gênero se compara às larguras dos intervalos de confiança para as proporções de gênero separadas?

# A largura da diferença de proporções será sempre maior que as larguras dos dois intervalos a partir dos quais ela é feita. Isso acontece devido à forma como o erro padrão é composto. Em geral, $\sqrt{A} < \sqrt{A + B}$ quando $B > 0$

# ## Questão 3
# 
# Construa um intervalo de 95% para a altura ([BMXHT](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.htm#BMXHT)) em centímetros. Em seguida, converta a altura de centímetros para polegadas dividindo por 2,54 e construa um intervalo de confiança de 95% para a altura em polegadas. Por fim, converta os pontos finais (os limites de confiança inferior e superior) do intervalo de confiança de polegadas para centímetros.

# In[21]:


# Primeiro pegamos os dados e olhamos para eles
height_data = da['BMXHT']
height_data.head()# Podemos calcular o intervalo de confiança de 95% usando os mesmos métodos de antes,
# desta vez usando apenas as fórmulas para uma média populacional

# Obtenha o desvio padrão da amostra
s = np.std(height_data, ddof = 1)
n = height_data.size

#Calcule o erro padrão
standard_error = s / np.sqrt(n)

# Calcule o multiplicador t para um intervalo de confiança de 95% (2,5% nas caudas)
t_multiplier = scipy.stats.t.ppf(q = 0.975, df = n - 1)

# Calcule o intervalo de confiança
x_hat = np.mean(height_data)
cm_lower_bound = x_hat - t_multiplier * standard_error
cm_upper_bound = x_hat + t_multiplier * standard_error
"Um IC para a altura, em centímetros, de alguém no estudo é: ({:.2f}, {:.2f})".format(
  cm_lower_bound,
  cm_upper_bound
)


# In[ ]:


# Podemos fazer o mesmo com a altura em polegadas
inches_height_data = height_data / 2.54

# Obtenha o desvio padrão da amostra
s = np.std(inches_height_data, ddof = 1)
n = inches_height_data.size

#Calcule o erro padrão
standard_error = s / np.sqrt(n)

# Calcule o multiplicador t para um intervalo de confiança de 95% (2,5% nas caudas)
t_multiplier = scipy.stats.t.ppf(q = 0.975, df = n - 1)

# Calcule o intervalo de confiança
x_hat = np.mean(inches_height_data)
inches_lower_bound = x_hat - t_multiplier * standard_error
inches_upper_bound = x_hat + t_multiplier * standard_error
"O IC para a altura, em polegadas, de alguém no estudo é: ({:.2f}, {:.2f})".format(
  inches_lower_bound,
  inches_upper_bound
)


# In[ ]:


# Se convertermos as medidas entre unidades, nossas respostas não deverão mudar
"O limite inferior em cm ({:.2f}) convertido em polegadas é {:.2f} (real = {:.2f})".format(
    cm_lower_bound, 
    cm_lower_bound / 2.54, 
    inches_lower_bound
)


# In[ ]:


"O limite superior em cm ({:.2f}) convertido em polegadas é {:.2f} (real = {:.2f})".format(
    cm_upper_bound, 
    cm_upper_bound / 2.54, 
    inches_upper_bound
)


# __Q3a.__ Descreva como o intervalo de confiança construído em centímetros se relaciona com o intervalo de confiança construído em polegadas.

# Eles são iguais. As unidades de uma medida não devem impactar a forma como a inferência estatística sobre essa quantidade é realizada para intervalos de confiança como estes

# ## Pergunta 4
# 
# Divida a amostra com base em faixas etárias de 10 anos, ou seja, os grupos resultantes serão compostos por pessoas com idades entre 18-28, 29-38, etc. Construa intervalos de confiança de 95% para a diferença entre o IMC médio para mulheres e para homens dentro cada faixa etária.

# In[ ]:


# Obtenha todas as faixas etárias usando a função de corte
age_ranges = pd.cut(da['RIDAGEYR'], bins = [0, 10, 20, 30, 40, 50, 60, 70, 80])
age_ranges.head()


# In[ ]:


# Obtenha uma lista exclusiva de todas as faixas etárias:
unique_age_ranges = list(set(age_ranges))
unique_age_ranges


# In[ ]:


# Primeiro vamos construir uma função para criar o intervalo de confiança de uma média.
# Esta fórmula pode ser complicada, então não se preocupe se precisar de um pouco de tempo
#para entender
def MakeUnPooledMeanConfidenceInterval(data_one, data_two, confidence):
    n1 = data_one.size
    n2 = data_two.size
    
    v1 = np.var(data_one, ddof = 1)
    v2 = np.var(data_two, ddof = 1)
    
    standard_error = np.sqrt(v1 / n1 + v2 / n2)
    
    t_multiplier = scipy.stats.t.ppf(1 - (1 - confidence) / 2, df = min(n1, n2))
    
    diff = np.mean(data_one) - np.mean(data_two)
    
    lower_bound = diff - t_multiplier * standard_error
    upper_bound = diff + t_multiplier * standard_error
    
    return {"n1": n1, "n2": n2, "lower_bound": lower_bound, "upper_bound": upper_bound}


# In[ ]:


# Agora vamos percorrer cada uma das faixas etárias, criar nossos dataframes e
# calcule os intervalos de confiança!
for age_range in sorted(unique_age_ranges):
    is_this_age = da[age_ranges == age_range]
    males_bmi = is_this_age[is_this_age['RIAGENDR'] == 1]['BMXBMI']
    females_bmi = is_this_age[is_this_age['RIAGENDR'] == 2]['BMXBMI']
    ci = MakeUnPooledMeanConfidenceInterval(males_bmi, females_bmi, 0.95)
    print("Idade: {} | IC para diferença em homens (n1:{}) imc menos mulheres (n2:{}) imc: ({:.2f}, {:.2f}) | Largura: {:.2f}".format(
        age_range, 
        ci["n1"],
        ci["n2"],
        ci['lower_bound'], 
        ci['upper_bound'], 
        ci['upper_bound'] - ci['lower_bound']
    ))


# __Q4a.__ Como as larguras desses intervalos de confiança diferem? Forneça uma explicação para quaisquer diferenças substanciais nas larguras dos intervalos de confiança que você vê.

# Esses intervalos de confiança podem variar por vários motivos. A principal razão pela qual eles variam é que os tamanhos de amostra de algumas partições são drasticamente maiores do que outros. Por exemplo, na faixa etária de 10 a 20 anos, temos apenas 175 homens e 165 mulheres. Compare isso com o restante das partições que geralmente possuem 400-500 amostras. Essa diferença no tamanho da amostra faz uma grande diferença no erro padrão e também no intervalo resultante.

# ## Pergunta 5
# 
# Construa um intervalo de confiança de 95% para a primeira e a segunda medidas de pressão arterial sistólica e para a diferença entre a primeira e a segunda medidas de pressão arterial sistólica dentro de um sujeito.

# In[ ]:


# Primeiro podemos preparar os dados extraindo as variáveis ​​do nosso dataframe. 
first_systolic = da['BPXSY1']
second_systolic = da['BPXSY2']

# Calcule as diferenças
diff_systolic = first_systolic - second_systolic


# In[ ]:


# Já fizemos a matemática longa acima, então vou criar uma função para fazer isso
# mais sistematicamente
def ComputeMeanConfidenceInterval(data, confidence):   
    s = np.std(data, ddof = 1)
    n = data.size
    standard_error = s / np.sqrt(n)
    t_multiplier = scipy.stats.t.ppf(q = 1 - (1 - confidence) / 2, df = n - 1)

    # Calcule o intervalo de confiança
    x_hat = np.mean(data)
    lower_bound = x_hat - t_multiplier * standard_error
    upper_bound = x_hat + t_multiplier * standard_error
    return {"lower_bound": lower_bound, "upper_bound": upper_bound, "n": n}

# Crie uma função realmente simples para imprimir o intervalo de confiança dos nossos dados
def PrintCIForSystolicMeasures(input_map, name_of_data):
    print("O intervalo de confiança para os dados {} é: ({:.2f}, {:.2f})".format(
        name_of_data, 
        input_map["lower_bound"], 
        input_map["upper_bound"]
    ))


# In[ ]:


# Calcula o IC para as medidas first_systolic e second_systolic
first_ci = ComputeMeanConfidenceInterval(first_systolic, confidence = 0.95)
PrintCIForSystolicMeasures(first_ci, "First Systolic")
second_ci = ComputeMeanConfidenceInterval(second_systolic, confidence = 0.95)
PrintCIForSystolicMeasures(second_ci, "Second Systolic")
# Temos sorte: a matemática para um intervalo de confiança de diferença média é o
# igual à matemática para um intervalo médio de uma população para que possamos reutilizar a função
# acima
diff_ci = ComputeMeanConfidenceInterval(diff_systolic, confidence = 0.95)
PrintCIForSystolicMeasures(diff_ci, "a diferença na primeira e segunda sistólica")


# __Q5a.__ Com base nesses intervalos de confiança, você diria que uma diferença de zero entre os valores médios da população da primeira e da segunda medidas de pressão arterial sistólica é consistente com os dados?

# A partir de nossos dados, estimamos com 95% de confiança que a diferença média da população entre a primeira e a segunda medidas de pressão arterial sistólica está entre 0,54 e 0,81 unidades. Como zero não está dentro deste intervalo, temos evidências suficientes para rejeitar a afirmação, com nível de confiança de 95%, de que a diferença média da população entre a primeira e a segunda medidas de pressão arterial sistólica é zero.

# __Q5b.__ Discuta como a largura do intervalo de confiança para a diferença dentro do sujeito se compara às larguras dos intervalos de confiança para a primeira e segunda medidas.

# A variação dentro do sujeito (a diferença entre as duas medidas) é menor que a variação intrapopulacional (a variação dentro da própria primeira medida e a variação dentro da própria segunda medida). Isto é, uma pessoa é mais parecida consigo mesma do que com a população de primeira e segunda medidas.

# # Pergunta 6
# 
# Construa um intervalo de confiança de 95% para a diferença média entre a idade média de um fumante e a idade média de um não fumante.

# In[ ]:


# Podemos preparar nossos dados subdividindo fumantes e não fumantes e, em seguida, extraindo suas idades
age_smokers = da[da['SMQ020'] == 1]['RIDAGEYR']
age_non_smokers = da[da['SMQ020'] == 2]['RIDAGEYR']

# Vamos reciclar nossa função acima para garantir que não cometeremos erros bobos!
diff_age_ci = MakeUnPooledMeanConfidenceInterval(age_smokers, age_non_smokers, confidence = 0.95)
"Um IC para a diferença de idade entre fumantes e não fumantes é: ({:.2f}, {:.2f}) anos".format(
    diff_age_ci["lower_bound"], 
    diff_age_ci["upper_bound"]
)


# __Q6a.__ Use técnicas gráficas e numéricas para comparar a variação nas idades dos fumantes com a variação nas idades dos não fumantes.

# In[ ]:


# Crie um histograma para fumantes
sns.histplot(age_smokers)
print("A média de idade dos fumantes é: {:.2f}".format(np.mean(age_smokers)))
print("O desvio padrão da idade dos fumantes é: {:.2f}".format(np.std(age_smokers, ddof = 1)))


# In[ ]:


# Crie um histograma para não fumantes
sns.histplot(age_non_smokers)
print("A média de idade dos fumantes é: {:.2f}".format(np.mean(age_non_smokers)))
print("O desvio padrão da idade dos fumantes é: {:.2f}".format(np.std(age_non_smokers, ddof = 1)))


# __Q6b.__ Parece que a incerteza sobre a idade média dos fumadores, ou a incerteza sobre a idade média dos não fumadores contribuiu mais para a incerteza da diferença média que estamos a focar aqui?

# O desvio padrão dos não fumadores é maior do que o desvio padrão dos fumadores, o que estaria a contribuir mais do que a incerteza para os fumadores. Os fumantes, em média, são mais velhos
