import os
import glob
import cv2 as cv

class Graph:
 
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    def addRelation(self, v, w):
        v = ord(v) - 65
        w = ord(w) - 65

        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

def rearrange_groups(old_groups):
    g = Graph(26)
    for grp in old_groups:
        u = grp[0]
        for x in range(1,len(grp)):
            # print(u,grp[x])
            g.addRelation(u,grp[x])

    cc = g.connectedComponents()

    new_group = []    
    for ngrp in cc:
        grp = ""
        for x in ngrp:
            grp = grp + chr(x+65)
        if(len(grp) > 1):
            new_group.append(grp)
    
    return new_group




similar_groups = []
letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
loc = '/content/drive/MyDrive/pre_data'
models = '/content/drive/MyDrive/models/'
from keras.models import model_from_json
json_file = open("/content/drive/MyDrive/models/model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("/content/drive/MyDrive/models/model-bw.h5")
# for i in letter:
#    print(i)

for i in letter:
  main_res = [0.]*27
  name = ""
  pt = '/content/drive/MyDrive/pre_data/' + i + '/*.jpg'
  path = glob.glob(pt)
  for img in path:
    image = cv.imread(img,0)
    test_image = cv.resize(image,(128,128))
    result = loaded_model.predict(test_image.reshape(1,128,128,1))
    print(result)
    for k in range(1,27):
      main_res[k] = main_res[k]+result[0][k]
  for j in range(1,27):
    main_res[j] = main_res[j]/3
    if(main_res[j] >= 0.3):
      name = name + letter[j-1]
  if len(name)>=2:
    similar_groups.append(name)
print(similar_groups)

new_groups = rearrange_groups(similar_groups)
print(new_groups)