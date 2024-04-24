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

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import scipy.stats

da = pd.read_csv("nhanes_2015_2016.csv")


# ## Questão 1
# Restrinja a amostra para mulheres entre 35 e 50 anos, então use a variável de estado civil DMDMARTL para dividir essa amostra em dois grupos - mulheres que estão atualmente casadas e mulheres que não estão atualmente casadas. Dentro de cada um desses grupos, calcule a proporção de mulheres que completaram o ensino superior e calcule intervalos de confiança de 95% para cada uma dessas proporções.

# In[ ]:


# First always inspect the data
da.head()


# In[ ]:


# Get the women between 35 and 50. The value of RIAGENDR is equal to 2 if a subject is female
middle_age_women = da.query('RIDAGEYR >= 35 & RIDAGEYR <= 50 & RIAGENDR == 2')

# Check that our data are correct! This step is not necessary but it is good practice :)
assert np.all(middle_age_women['RIAGENDR'] == 2)
assert np.all(middle_age_women['RIDAGEYR'] <= 50)
assert np.all(middle_age_women['RIDAGEYR'] >= 35)


# In[ ]:


# Partition the group into married and non-married women
married_middle_age_women = middle_age_women.query('DMDMARTL == 1')
non_married_middle_age_women = middle_age_women.query('DMDMARTL != 1')

# More data checks!
assert np.all(married_middle_age_women['DMDMARTL'] == 1)
assert np.all(non_married_middle_age_women['DMDMARTL'] != 1)


# In[ ]:


# We now have checked our data is valid and is partitioned correctly. We 
# now can compute the 90% confidence intervals for the proportion which have
# completed college. This is coded in the DMDEDUC2 variable and it is equal to
# 5 if they have completed college or above
married_and_completed_college = married_middle_age_women['DMDEDUC2'] == 5
p_hat_married = np.mean(married_and_completed_college)
married_sample_size = married_and_completed_college.size
"The proportion of married middle age women (N={}) who completed college is: {:.2f}".format(
    married_sample_size, 
    p_hat_married
)


# In[ ]:


# We can do the same for non-married women
non_married_and_completed_college = non_married_middle_age_women['DMDEDUC2'] == 5
p_hat_non_married = np.mean(non_married_and_completed_college)
non_married_sample_size = non_married_and_completed_college.size
"The proportion of non-married middle age women (N={}) who completed college is: {:.2f}".format(non_married_sample_size, p_hat_non_married)


# In[ ]:


# We can now compute the confidence intervals. Remember, for a two-sided 
# confidence interval, we need 5% in each of the tails and 95% PPF will give us
# this value !
z_multiplier = scipy.stats.norm.ppf(q = 0.95)
married_standard_error = np.sqrt(p_hat_married * (1 - p_hat_married) / married_sample_size)
ci_married_lower_bound = p_hat_married - z_multiplier * married_standard_error
ci_married_upper_bound = p_hat_married + z_multiplier * married_standard_error
"A 90% confidence interval for the proportion of married women who completed college is ({:.2f}, {:.2f})".format(
    ci_married_lower_bound, 
    ci_married_upper_bound
)


# In[ ]:


# We now can do the same for non-married women
z_multiplier = scipy.stats.norm.ppf(q = 0.95)
non_married_standard_error = np.sqrt(p_hat_non_married * (1 - p_hat_non_married) / non_married_sample_size)
ci_non_married_lower_bound = p_hat_non_married - z_multiplier * non_married_standard_error
ci_non_married_upper_bound = p_hat_non_married + z_multiplier * non_married_standard_error
"A 90% confidence interval for the proportion of non-married women who completed college is ({:.2f}, {:.2f})".format(ci_non_married_lower_bound, ci_non_married_upper_bound)


# __Q1a.__ Identifique qual dos dois intervalos de confiança é mais amplo e explique por que isso acontece.

# A largura do intervalo de confiança para a proporção populacional de mulheres casadas que completaram o ensino superior é de oito pontos percentuais, enquanto o intervalo de confiança para mulheres não casadas é de sete pontos percentuais. O intervalo de confiança para mulheres casadas é maior, apesar de ter um tamanho de amostra maior, porque a estimativa $\hat{p}{casadas}$ está mais próxima de 50% do que $\hat{p}{não-casadas}$, então o erro padrão de $\hat{p}_{casadas}$ é maior.

# __Q1b.__ Escreva uma ou duas frases resumindo essas descobertas para um público que não sabe o que é um intervalo de confiança (o objetivo aqui é relatar a substância do que você aprendeu sobre como o estado civil e o nível de escolaridade estão relacionados, e não ensinar a uma pessoa o que é um intervalo de confiança). intervalo de confiança é).

# Podemos ver que, em média, uma mulher de meia-idade casada tem maior probabilidade de ter concluído a faculdade do que uma mulher de meia-idade que não é casada, e os intervalos de confiança para estas duas estimativas não se sobrepõem.

# ## Questão 2
# 
# Construa um intervalo de confiança de 95% para a proporção de mulheres fumantes. Construa um intervalo de confiança de 95% para a proporção de fumantes do sexo masculino. Construa um intervalo de confiança de 95% para a **diferença** entre essas duas proporções de gênero.

# In[ ]:


# First, lets prepare our data
females = da.query('RIAGENDR == 2')
males = da.query('RIAGENDR == 1')

# I found the smoking variable here: https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/SMQ_I.htm#SMQ020
is_female_smoker = females['SMQ020'] == 1
is_male_smoker = males['SMQ020'] == 1


# In[ ]:


# In order to reduce the chance of mistakes, I'm going to create a simple function
# for computing confidence intervals. This function will do the exact same steps as 
# above so nothing should be scary :)
def MakeAConfidenceInterval(data, confidence):
    p_hat = np.mean(data)
    n = data.size
    z_multiplier = scipy.stats.norm.ppf(q = 1 - (1 - confidence) / 2)
    standard_error = np.sqrt(p_hat * (1 - p_hat) / n)
    lower_bound = p_hat - z_multiplier * standard_error
    upper_bound = p_hat + z_multiplier * standard_error
    return {"phat": p_hat, "n":n, "lower_bound":lower_bound, "upper_bound":upper_bound}


# In[ ]:


female_smoking_ci = MakeAConfidenceInterval(data = is_female_smoker, confidence = 0.95)
"The CI for female smokers (N={}) is ({:.2f}, {:.2f})".format(
    female_smoking_ci['n'],
    female_smoking_ci['lower_bound'],
    female_smoking_ci['upper_bound']
)


# In[ ]:


male_smoking_ci = MakeAConfidenceInterval(data = is_male_smoker, confidence = 0.95)
"A CI for male smokers (N={}) is ({:.2f}, {:.2f})".format(
    male_smoking_ci['n'],
    male_smoking_ci['lower_bound'],
    male_smoking_ci['upper_bound']
)


# In[ ]:


# Now we can construct the confidence interval for the difference between the proportions.
# This formula is pretty long so we will break it down into parts

# First compute the standard error of the difference
standard_error = np.sqrt(
    female_smoking_ci['phat'] * (1 - female_smoking_ci['phat']) / female_smoking_ci['n'] + 
    male_smoking_ci['phat'] * (1 - male_smoking_ci['phat']) / male_smoking_ci['n']
)

# Get the z-multiplier for our 95% confidence interval
z_multiplier = scipy.stats.norm.ppf(q = 0.975)

# Compute the point estimate for the differences between the male and female smoking proportion
difference = male_smoking_ci['phat'] - female_smoking_ci['phat']

# Compute the lower_bound and the upper_bound of the confidence interval
lower_bound = difference - z_multiplier * standard_error
upper_bound = difference + z_multiplier * standard_error
"A CI for the difference between male smoking proportion less the female smoking proportion is ({:.2f}, {:.2f})".format(
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

# In[ ]:


# First we get the data and look at it
height_data = da['BMXHT']
height_data.head()# We can compute the 95% confidence interval using the same methods as before, 
# this time just using the formulas for a one population mean

# Get the sample standard deviation
s = np.std(height_data, ddof = 1)
n = height_data.size

# Compute the standard error
standard_error = s / np.sqrt(n)

# Compute the t-multiplier for a 95% confidence interval (2.5% in the tails)
t_multiplier = scipy.stats.t.ppf(q = 0.975, df = n - 1)

# Compute the confidence interval
x_hat = np.mean(height_data)
cm_lower_bound = x_hat - t_multiplier * standard_error
cm_upper_bound = x_hat + t_multiplier * standard_error
"A CI for the height, in centimeters, of someone in the study is: ({:.2f}, {:.2f})".format(
  cm_lower_bound,
  cm_upper_bound
)


# In[ ]:


# We can do the same thing for height in inches

inches_height_data = height_data / 2.54

# Get the sample standard deviation
s = np.std(inches_height_data, ddof = 1)
n = inches_height_data.size

# Compute the standard error
standard_error = s / np.sqrt(n)

# Compute the t-multiplier for a 95% confidence interval (2.5% in the tails)
t_multiplier = scipy.stats.t.ppf(q = 0.975, df = n - 1)

# Compute the confidence interval
x_hat = np.mean(inches_height_data)
inches_lower_bound = x_hat - t_multiplier * standard_error
inches_upper_bound = x_hat + t_multiplier * standard_error
"The CI for the height, in inches, of someone in the study is: ({:.2f}, {:.2f})".format(
  inches_lower_bound,
  inches_upper_bound
)


# In[ ]:


# If we convert the measurements back and forth between units, our answers should not change
"The cm lower bound ({:.2f}) converted to inches is {:.2f} (actual = {:.2f})".format(
    cm_lower_bound, 
    cm_lower_bound / 2.54, 
    inches_lower_bound
)


# In[ ]:


"The cm upper bound ({:.2f}) converted to inches is {:.2f} (actual = {:.2f})".format(
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


# Get all of the ranges of age using the cut function
age_ranges = pd.cut(da['RIDAGEYR'], bins = [0, 10, 20, 30, 40, 50, 60, 70, 80])
age_ranges.head()


# In[ ]:


# Get a unique list of all of the age ranges:
unique_age_ranges = list(set(age_ranges))
unique_age_ranges


# In[ ]:


# First we are going to construct a function for creating the confidence interval of a mean.
# This formula can be complicated so don't worry if you have to take a little bit of time
# to understand it
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


# Now we are going to loop through each of the age ranges, make our dataframes, and 
# compute the confidence intervals !
for age_range in sorted(unique_age_ranges):
    is_this_age = da[age_ranges == age_range]
    males_bmi = is_this_age[is_this_age['RIAGENDR'] == 1]['BMXBMI']
    females_bmi = is_this_age[is_this_age['RIAGENDR'] == 2]['BMXBMI']
    ci = MakeUnPooledMeanConfidenceInterval(males_bmi, females_bmi, 0.95)
    print("Age: {} | CI for difference in male (n1:{}) bmi less female (n2:{}) bmi: ({:.2f}, {:.2f}) | Width: {:.2f}".format(
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


# First we can prep the data by extracting out the variables from our dataframe. 
first_systolic = da['BPXSY1']
second_systolic = da['BPXSY2']

# Compute the differences
diff_systolic = first_systolic - second_systolic


# In[ ]:


# We already did the long form math above so I'm going to create a function to do this
# more systematically
def ComputeMeanConfidenceInterval(data, confidence):   
    s = np.std(data, ddof = 1)
    n = data.size
    standard_error = s / np.sqrt(n)
    t_multiplier = scipy.stats.t.ppf(q = 1 - (1 - confidence) / 2, df = n - 1)

    # Compute the confidence interval
    x_hat = np.mean(data)
    lower_bound = x_hat - t_multiplier * standard_error
    upper_bound = x_hat + t_multiplier * standard_error
    return {"lower_bound": lower_bound, "upper_bound": upper_bound, "n": n}

# Create a really simple function for printing out the confidence interval of our data
def PrintCIForSystolicMeasures(input_map, name_of_data):
    print("The confidence interval for the {} data is: ({:.2f}, {:.2f})".format(
        name_of_data, 
        input_map["lower_bound"], 
        input_map["upper_bound"]
    ))


# In[ ]:


# Compute the CI for the first_systolic and second_systolic measures
first_ci = ComputeMeanConfidenceInterval(first_systolic, confidence = 0.95)
PrintCIForSystolicMeasures(first_ci, "First Systolic")
second_ci = ComputeMeanConfidenceInterval(second_systolic, confidence = 0.95)
PrintCIForSystolicMeasures(second_ci, "Second Systolic")
# We are fortunate: the math for a mean difference confidence interval is the 
# same as the math for a one-population mean interval so we can reuse the function
# above
diff_ci = ComputeMeanConfidenceInterval(diff_systolic, confidence = 0.95)
PrintCIForSystolicMeasures(diff_ci, "the difference in the first and second systolic")


# __Q5a.__ Com base nesses intervalos de confiança, você diria que uma diferença de zero entre os valores médios da população da primeira e da segunda medidas de pressão arterial sistólica é consistente com os dados?

# A partir de nossos dados, estimamos com 95% de confiança que a diferença média da população entre a primeira e a segunda medidas de pressão arterial sistólica está entre 0,54 e 0,81 unidades. Como zero não está dentro deste intervalo, temos evidências suficientes para rejeitar a afirmação, com nível de confiança de 95%, de que a diferença média da população entre a primeira e a segunda medidas de pressão arterial sistólica é zero.

# __Q5b.__ Discuta como a largura do intervalo de confiança para a diferença dentro do sujeito se compara às larguras dos intervalos de confiança para a primeira e segunda medidas.

# A variação dentro do sujeito (a diferença entre as duas medidas) é menor que a variação intrapopulacional (a variação dentro da própria primeira medida e a variação dentro da própria segunda medida). Isto é, uma pessoa é mais parecida consigo mesma do que com a população de primeira e segunda medidas.

# # Pergunta 6
# 
# Construa um intervalo de confiança de 95% para a diferença média entre a idade média de um fumante e a idade média de um não fumante.

# In[ ]:


# We can prepare our data by subsetting for smokers and non-smokers and then extacting their age
age_smokers = da[da['SMQ020'] == 1]['RIDAGEYR']
age_non_smokers = da[da['SMQ020'] == 2]['RIDAGEYR']

# We are going to recycle our function from above to make sure we don't make any silly mistakes! 
diff_age_ci = MakeUnPooledMeanConfidenceInterval(age_smokers, age_non_smokers, confidence = 0.95)
"A CI for the age difference between smokers less non-smokers is: ({:.2f}, {:.2f}) years".format(
    diff_age_ci["lower_bound"], 
    diff_age_ci["upper_bound"]
)


# __Q6a.__ Use técnicas gráficas e numéricas para comparar a variação nas idades dos fumantes com a variação nas idades dos não fumantes.

# In[ ]:


# Create a histogram for smokers
sns.histplot(age_smokers)
print("The mean of smokers age is: {:.2f}".format(np.mean(age_smokers)))
print("The standard deviation of smokers age is: {:.2f}".format(np.std(age_smokers, ddof = 1)))


# In[ ]:


# Create a histogram for nonsmokers
sns.histplot(age_non_smokers)
print("The mean of smokers age is: {:.2f}".format(np.mean(age_non_smokers)))
print("The standard deviation of smokers age is: {:.2f}".format(np.std(age_non_smokers, ddof = 1)))


# __Q6b.__ Parece que a incerteza sobre a idade média dos fumadores, ou a incerteza sobre a idade média dos não fumadores contribuiu mais para a incerteza da diferença média que estamos a focar aqui?

# O desvio padrão dos não fumadores é maior do que o desvio padrão dos fumadores, o que estaria a contribuir mais do que a incerteza para os fumadores. Os fumantes, em média, são mais velhos
