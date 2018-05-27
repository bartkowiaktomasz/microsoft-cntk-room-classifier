import csv
import os as os
import numpy as np
#traversal the path
def traversal(fpath):
    #check whether the path is exist
    if os.path.exists(fpath) == False:
        print ("no such dir")
        return False
    #get the filename & filter unwanted file
    #return a list of object in each room & return the filename& dir name
    oblist=[]
    roomroot=[]
    roomfile=[]
    for (root, dirs, files) in os.walk(fpath):  
        for filename in files:
            if filename.endswith(exts):
                newpath=os.path.join(root,filename)
                oblist.append(readFile(newpath))
                roomroot.append(root)
                roomfile.append(filename)
    return oblist,roomroot,roomfile
              
        
#read the file
def readFile(filename):
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    oblist=[]
    for row in csv_reader: 
        oblist.append(row[0].split()[0])
    #print(oblist)
    return oblist

def ETL(obmap,oblist,roomtype,filename):
    result=np.zeros((len(oblist),len(obmap)), dtype=int)
    for row in range(0,len(oblist)):
        for objects in oblist[row]:
            for i in range(1,len(obmap)):
                if objects == obmap[i]:
                    result[row][i] +=1
    roomtype=np.array(roomtype).reshape((len(oblist),1))
    filename=np.array(filename).reshape((len(oblist),1))
    return result,roomtype,filename

def convert_to_room_type(roomroot):
    path_list = roomroot.tolist()
    roomtype = []
    for value in path_list:
        index = value[0].index('/') + 1
        type = value[0][index:]
        roomtype.append(type)
    return roomtype

def result_vector_creation(type_result, all_types):
    results = []
    for value in type_result:
        index = all_types.index(value)
        results.append(index)
    return results

def save_data(data, filename):
    data.dump(filename)

def load_data(filename):
    x = np.load(filename)
    return x

def remapping(x):
    result = x
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j]>=2:
                result[i][j] = 1
    return result

#main
#initial the path
fpath = 'testImages'
exts = 'labels.tsv'
mappath ='class_map.txt'
room_type_path = 'room_types.txt'
#load the map
obmap = readFile(mappath)
room_map = readFile(room_type_path)
oblist,roomroot,roomfile = traversal(fpath)
#ETL
objects_result,roomroot,roomfile = ETL(obmap,oblist,roomroot,roomfile)
roomtype = convert_to_room_type(roomroot)
types_result = result_vector_creation(roomtype, room_map)
# convert the list into array
types_result = np.asarray(types_result)
print(objects_result)
print(types_result)

# remap the matrix, and now the matrix is a binary matrix
objects_result = remapping(objects_result)

save_data(objects_result, "object_result.dat")
save_data(types_result, "types_result.dat")
# x = load_data("object_result.dat")
