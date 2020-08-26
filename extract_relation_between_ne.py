
import pandas as pd
import os
import nltk
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

def get_useful_sentences_tmp(file_path):
    data_paths=os.listdir(file_path)
    df=pd.DataFrame()
    for f in data_paths:
        sub=pd.read_csv(file_path+f,header=None)
        sub=sub[sub.columns[sub.count().values>12][-2:]]
        sub.columns=[0,1]
        sub=sub[sub[1]==1]
        if not df.shape[0]:
            df=sub
        else:
            df=pd.concat([df,sub],axis=0)
    text=list(df[0])
    text=" ".join(text)
    sentences = nltk.sent_tokenize(text)
    return sentences



def get_filtered_text(file_path):
    import os
    import nltk
    data_paths=os.listdir(file_path)
    text=""
    for t in data_paths:
        with open (file_path+t,"r") as f:
            t=f.read()
            text=text+" "+t
    sentences = nltk.sent_tokenize(text)
    return sentences


def contain(ls1,ls2):
    for i in ls1:
        if i in ls2:
            return True
    else:
        return False

def shortest_dependency_path(doc, ne_list,e1, e2,stoplist = ['prep']):
    """
    function to get the shortest dependency path between two named entities in a sentence
    :param doc: annotated sentence
    :param ne_list: named entity list
    :param e1: named entity 1
    :param e2: named entity 2
    :param stoplist: stop list
    :return: triple - (e1,relation,e2)
    """
    import networkx as nx
    import string
    edges = []
    words_to_remove=[]
    for token in doc:
        for child in token.children:
            if child.dep_ in stoplist:
                words_to_remove.append(child.text)
            for ne in ne_list:
                if str(token) in ne.split(" "):
                    token = ne
                if str(child) in ne.split(" "):
                    child = ne
            
            edges.append(('{0}'.format(token),
                          '{0}'.format(child)))
            
    graph = nx.Graph(edges)
    try:
        shortest_path = nx.shortest_path(graph, source=e1, target=e2)
    except:
        shortest_path = []
    shortest_path=[w for w in shortest_path if w not in words_to_remove and w not in string.punctuation]
    if len(shortest_path)>=3 and len(shortest_path)<7 :
        relation=" ".join(shortest_path[1:-1])
        return (e1,relation,e2)
    else:
        return None


def is_entity(candidate_words):
    """
    function to check whether the named entity returned by spaCy is a named entity in wikipedia
    :param candidate_ngram:
    :return: link,title,entity_type
    """
    import requests
    from bs4 import BeautifulSoup
    url = 'https://en.wikipedia.org/wiki/' + '%20'.join(candidate_words)
    try:        
        page = BeautifulSoup(requests.get(url).content,'html.parser')
        if 'Wikipedia does not have an article with this exact name' in page.get_text():
            return None,None,None
        if 'ask for it to be created' in page.get_text():
            return None,None,None
        elif page.find('a',{'href':'/wiki/Category:Disambiguation_pages'}):
            return None,None,None
        else:
            
            link = page.find('link',{'rel':'canonical'}).get('href')
            title = page.find('h1',id="firstHeading").get_text()
            entity_type = 'UNKNOWN'
            if page.find('a',{'href':'/wiki/Category:Living_people'}) or page.find('table',{'class':'biography'}):
                entity_type = "PERSON"
            return link,title,entity_type
    except Exception as e:
        print(e)
        return None,None,None    


def get_triple_text_pairs(sentences):
    """
    Get triples from the corpus
    :param sentences:
    :return: triples, entity type, shortest dependency path, and sentence
    """
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    triples=[]
    triple_text_pairs=[]
    types = ['ORG',  'GPE', 'PERSON', 'NORP', 'FAC', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
    for sentence in sentences:
        doc = nlp_spacy(sentence) 
        ne_list=[]    
        ne_label_dict={}
        for ent in doc.ents:
            if not ent.label_ in types:
                continue
            candidate_text = ent.text.strip()
            link, title,entity_type = is_entity(nltk.word_tokenize(candidate_text))
            if link:
                if entity_type == 'UNKNOWN':
                    entity_type = ent.label_
                sentence = sentence.replace(candidate_text,title)
                ne_list.append(title)
                ne_label_dict[title]=entity_type  

        if len(ne_list)<2:
            continue        
        ne_list_set=list(set(ne_list))
        doc = nlp_spacy(sentence)
        for n in range(len(ne_list_set)-1):
            subj=ne_list_set[n]
            for obj in ne_list_set[n+1:]:
                triple=shortest_dependency_path(doc,ne_list,subj,obj)
                if triple:
                    triples.append(triple)
                    triple_text_pairs.append([*triple,ne_label_dict[subj],ne_label_dict[obj],sentence])
    return triple_text_pairs


def convert_to_df(triple_text_pairs,doc_level=False):
    """
    :param triple_text_pairs:
    :param doc_level: if true, look at the whole corpus to identify the relationship between two entities; if
    false, only look at the single sentence.
    :return: a dataframe
    """
    import pandas as pd
    if doc_level:
        triple_text_pairs_set={}
        for t in triple_text_pairs:
            if not triple_text_pairs_set.get((t[0],t[2])):
                triple_text_pairs_set[(t[0],t[2])]=[t[3],t[4],t[1],t[5]]
            else:
                ori=triple_text_pairs_set[(t[0],t[2])]
                triple_text_pairs_set[(t[0],t[2])]=[ori[0],ori[1],ori[2]+" "+t[1],ori[3]+" "+t[5]]
        triple_text=[]
        for k,v in triple_text_pairs_set.items():
            triple_text.append([k[0],v[2],k[1],v[0],v[1],v[3]])
    else:
        triple_text=triple_text_pairs
    
    df=pd.DataFrame(triple_text)
    df.columns=["subject","dependency_path","object","subj_type","obj_type","sentence"]
    return df


def encode_entity_type(df):
    """
    One hot encode the type of the named entities

    """
    from sklearn import preprocessing
    types = ['ORG',  'GPE', 'PERSON', 'NORP', 'FAC', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
    types.append("OOT")
    le = preprocessing.LabelEncoder()
    le=le.fit(types)
    le_types=le.transform(types)
    le_types_arr=np.reshape(le_types,(-1,1))

    one_hot_e = preprocessing.OneHotEncoder()
    one_hot_e=one_hot_e.fit(le_types_arr)

    type_sub = df["subj_type"].values
    type_sub_arr=le.transform(type_sub)
    type_sub_arr=np.reshape(type_sub_arr,(-1,1))
    type_sub_arr=one_hot_e.transform(type_sub_arr)

    type_obj = df["obj_type"].values
    type_obj_arr=le.transform(type_obj)
    type_obj_arr=np.reshape(type_obj_arr,(-1,1))
    type_obj_arr=one_hot_e.transform(type_obj_arr)

    type_sub_obj = df["subj_type"].str.cat(df["obj_type"]).values
    le2 = preprocessing.LabelEncoder()
    type_sub_obj_arr=le2.fit(type_sub_obj).transform(type_sub_obj)
    one_hot_e2 = preprocessing.OneHotEncoder()
    type_sub_obj_arr=np.reshape(type_sub_obj_arr,(-1,1))
    type_sub_obj_arr=one_hot_e2.fit(type_sub_obj_arr).transform(type_sub_obj_arr)
    
    return type_sub_arr,type_obj_arr,type_sub_obj_arr


def get_embeded_word_vectors(pretrained_model_path,df,dependency_weight, non_dependency_weight):
    """
    :param pretrained_model_path: the path to the pre-trained glove model
    :param df:
    :param dependency_weight: the weight we give to the words that appears in the shortest dependency path
    :param non_dependency_weight: the weight we give to the words that do not appear in the shortest dependency path
    :return: word vectors
    """
    from GloveVectorizer import GloveVectorizer
    glove_vectorizer = GloveVectorizer(pretrained_model_path)
    glove_vectorizer.fit(df["sentence"].values,df["dependency_path"].values,df,wgt=dependency_weight, wgt_inverse=non_dependency_weight)
    w2v = glove_vectorizer.transform_sumembed(df["sentence"], idf=True,weights=True)
    return w2v


def get_features(triple_text_pairs,pretrained_model_path="glove.6B/glove.6B.100d.txt",dependency_weight=1, non_dependency_weight=0.5):
    """
    :param triple_text_pairs:
    :param pretrained_model_path: the path to the pre-trained glove model
    :param dependency_weight: the weight we give to the words that appears in the shortest dependency path
    :param non_dependency_weight: the weight we give to the words that do not appear in the shortest dependency path
    :return: combined fetaures array
    """
    from sklearn.decomposition import PCA
    df=convert_to_df(triple_text_pairs,doc_level=False)
    type_sub_arr,type_obj_arr,type_sub_obj_arr=encode_entity_type(df)
    w2v=get_embeded_word_vectors(pretrained_model_path,df,dependency_weight, non_dependency_weight)

    tmp = []
    pca = PCA(18)
    w2v_pca=pca.fit_transform(w2v)
    tmp.append(w2v_pca)

    pca = PCA(6)
    type_obj_arr=pca.fit_transform(type_obj_arr.A)
    tmp.append(type_obj_arr)

    pca = PCA(6)
    type_sub_arr_pca=pca.fit_transform(type_sub_arr.A)
    tmp.append(type_sub_arr_pca)
    
    #pca = PCA(21)
    #type_sub_obj_arr_pca=pca.fit_transform(type_sub_obj_arr.A)
    tmp.append(type_sub_obj_arr.A)
    X = np.hstack(tmp)
    print(X.shape)
    return df,X


def tune_cluster(n_range=[]):
    score={}
    for n in n_range:
        clustering = AgglomerativeClustering(n_clusters=n,linkage="complete")
        pred = clustering.fit_predict(X)
        m=metrics.silhouette_score(X, pred)
        score[n]=m
    return score

def cluster(X,n_clusters):
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters,linkage="complete")
    pred = clustering.fit_predict(X)
    return pred


def get_relationship(df,pred):
    from nltk import word_tokenize
    pred_index=np.arange(pred.shape[0])
    cluster_repr_dic={}
    for i in set(pred):
        text_index=pred_index[pred==i]
        text=" ".join(df["dependency_path"][text_index].values)
        words=word_tokenize(text)
        word_cnt_dic={}
        for w in words:
            if word_cnt_dic.get(w):
                word_cnt_dic[w]=word_cnt_dic[w]+1
            else:
                word_cnt_dic[w]=1
        top_freq=sorted(word_cnt_dic.items(), key=operator.itemgetter(1),reverse=True)[:3]
        top_freq="-".join([str(i[0]) for i in top_freq])
        cluster_repr_dic[i]=top_freq
                           
    def get_rel(x):
        rel=cluster_repr_dic.get(x)
        return rel
    df["cluster"]=pred
    df["rel"]=df["cluster"].apply(lambda x:get_rel(x))

    return df



