# coding=UTF-8
'''
Created on 15 de out de 2017

@author: andre.nascimento

'''

import io
import os
import re
import sys
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from nltk import FreqDist
from nltk import tokenize
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import sklearn
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rn

def main():
    #caminho da pasta com o arquivo
    pasta="c:/users/usuario/dados"
    alterarPasta(pasta)
    nomeArquivo="twits_classificados.csv"
    df=lerCSV(nomeArquivo)
    
    
        #remover a coluna usuário
    dft=removerColuna(df)
    
    #divide o dataframe em tamanho inicial para treino e tamanho final para teste, gerando um intervalo aleatório entre (tamanhoInicial-tamanhoFinal)
    tamanhoInicial=100
    tamanhoFinal=100
    
    numero=dividirDataFrame(len(df), tamanhoInicial, tamanhoFinal)
    texto=df['texto'].iloc[numero].get_values()
    treino=etl(dft.head(tamanhoInicial))
    teste=etl(dft.tail(tamanhoFinal))
    
    classificacao=classificar1(treino, teste, texto)
    print(classificacao)

def alterarPasta(pasta):
    os.chdir(pasta)

def lerCSV(nomeArquivo):
    return pd.read_csv(nomeArquivo, encoding='ISO-8859-1', sep=";", header=0)

def removerPontuacao(texto):
    return re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', ' ', texto)

def removerStopWord(texto):
    stopWord=nltk.corpus.stopwords.words('portuguese')
    sentencaSemStopword = [i for i in texto.split() if not i in stopWord]
    return " ".join(sentencaSemStopword)

def aplicarTokenize(texto):
    return tokenize.word_tokenize(texto, language='portuguese')

def removerNumeros(texto):
    return re.sub('[0-9]', '', texto)

def removerURL(texto):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','', texto)

def transformarEmMinusculas(texto):
    return texto.lower()

def aplicarStemming(texto):
    texto=str(texto)
    stemmer = nltk.stem.RSLPStemmer()
    palavras=[]
    for txt in texto.split():
        palavra=stemmer.stem(txt)
        palavras.append(palavra)
    return " ".join(palavras)

def removerColuna(dataframe):
    dataframe=dataframe.drop(dataframe.columns[0], axis=1)
    return dataframe

def removerMensionamento(texto):
    if texto.find("@")>=0:
        texto=texto.split("@")
        texto=str(texto[-1])
        contem=texto.find(" ")
        texto=texto[contem+1:]
        
    return texto

def etl(df):
    
    twits=[]
    # pega apenas os valores do dataframe
    textos=df.values
 #trata o texto de cada linha do dataframe
    for texto in textos:
        
        texto[0]=str(texto[0])
        
        #remover mensionamento (@usuario)
        texto[0]=removerMensionamento(texto[0])
        
        #remove URL
        texto[0]=removerURL(texto[0])
        
        #ransforma a frase toda em minúscula
        texto[0]=transformarEmMinusculas(texto[0])
        
        #remove números
        texto[0]=removerNumeros(texto[0])
        
        #remove as pontuações
        texto[0]=removerPontuacao(texto[0])
        
        #retirar assentos
        #texto[0]=removerAssentuacao(texto[0])
        
        #aplica stemming
        texto[0]=aplicarStemming(texto[0])
        
        #remove stopwords
        texto[0]=removerStopWord(texto[0])
        
        texto[0]=str(texto[0])
        
        twits.append((texto[0], texto[1]))
        
    return twits

def classificar1(treino, teste, frase):
    texto=tratarTexto(frase)
    
    cl = NaiveBayesClassifier(treino)
    
    accuracy = cl.accuracy(teste)
    
    blob = TextBlob(texto,classifier=cl)
    
    frase=str(frase)
    return frase+' ({})'.format(blob.classify())+'\nacurácia: {}'.format(accuracy)

def tratarTexto(texto):
    texto=str(texto)
    
    #remover mensionamento (@usuario)
    texto=removerMensionamento(texto)
    
    #remove URL
    texto=removerURL(texto)
    
    #transforma a frase toda em minúscula
    texto=transformarEmMinusculas(texto)
    
    #remove números
    texto=removerNumeros(texto)
    
    #remove as pontuações
    texto=removerPontuacao(texto)
    
    #retirar assentos
    #texto=removerAssentuacao(texto)
    
    #aplica stemming
    texto=aplicarStemming(texto)
    
    #remove stopwords
    texto=removerStopWord(texto)
    
    texto=str(texto)
    
    return texto

def dividirDataFrame(tamanhoDF, tamanhoInicial, tamanhoFinal):
    tamanhoFinal=tamanhoFinal+1
    tamanhoFinal=tamanhoDF-tamanhoFinal
    return rn.sample(range(tamanhoInicial, tamanhoFinal), 1)
    
def classificar2(df, texto):
    #esta classificação é feita usando o NaiveBayes do skit-learn
    
    textos=df['texto'].values
    sentimento=df['sentimento'].values
    
    #a linha abaixo traz a o vetor de 1 palavra
    #vectorizer = CountVectorizer(analyzer="word")
    
    #Já esta abaixo, tráz de 2 em 2 palavras. Obs, o resultado desta foi melhor
    vectorizer = CountVectorizer(ngram_range=(1,2))
    freqWords = vectorizer.fit_transform(textos)
    modelo = MultinomialNB()
    modelo.fit(freqWords, sentimento)
    freq_testes = vectorizer.transform(texto)
    classificacao=modelo.predict(freq_testes)
    resultados = cross_val_predict(modelo, freqWords, sentimento, cv=10)
    acuracia=metrics.accuracy_score(sentimento, resultados)
    
    classes=["inseguro", "nêutro", "seguro"]
    #print(metrics.classification_report(sentimento, resultados, classes))
    print(pd.crosstab(sentimento, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')
    texto=str(texto)
    return texto+' ({})'.format(classificacao)+'\nacurácia: {}'.format(acuracia)

def calcularPercentualTotal(df):
    numero_seguros = len(df.loc[df['sentimento'] == 'seguro'])
    numero_inseguros= len(df.loc[df['sentimento'] == 'inseguro'])
    numero_neutros = len(df.loc[df['sentimento'] == 'nêutro'])
    
    print("Número de inseguros: {0} ({1:2.2f}%)".format(numero_inseguros, (numero_inseguros/(numero_inseguros+numero_seguros+numero_neutros))*100))
    print("Número de seguros: {0} ({1:2.2f}%)".format(numero_seguros, (numero_seguros/(numero_inseguros+numero_seguros+numero_neutros))*100))
    print("Número de nêutros: {0} ({1:2.2f}%)".format(numero_neutros, (numero_neutros/(numero_inseguros+numero_seguros+numero_neutros))*100))

