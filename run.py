from extract_relation_between_ne import *

filePath = "../ocp_data2/filtered_text_files/filter0/"
sentences=get_filtered_text(filePath)
triple_text_pairs = get_triple_text_pairs(sentences)
df,X = get_features(triple_text_pairs,pretrained_model_path="glove.6B/glove.6B.100d.txt",dependency_weight=1, non_dependency_weight=0.5)
scores=tune_cluster(n_range=np.arange(1,20)*10)
print(scores)

pred=cluster(X,n_clusters=30)
df_final=get_relationship(df,pred)


# Visualize the knowledge graph
from graphviz import Digraph
dot = Digraph(comment='test')
for i,a in df_final.iterrows():
    dot.edge(a[0],a[2],label=a[7])
dot

