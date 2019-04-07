import numpy as np

filename = "data1.txt"
a_list0 = []
b_list0 = []
c_list0 = []
a_list = []
b_list = []
c_list = []
def str_2_list(data_list):
    ret_list = []
    for i in range(len(data_list)):
        tmp_list = data_list[i].strip().split(" ")
        tmp_ret_list = [int(tmp_list[7][0]),int(tmp_list[6]),int(tmp_list[5]),int(tmp_list[4]),int(tmp_list[3]),int(tmp_list[2]),int(tmp_list[1]),int(tmp_list[0][1])]
        ret_list.append(tmp_ret_list)
    return ret_list

with open(filename, "r") as file:
    filein = file.read().splitlines()
    for item in filein:
        tmp_list = item.strip().split(",")
        a_list0.append(tmp_list[0])
        b_list0.append(tmp_list[1])
        c_list0.append(tmp_list[2])
a_list0 = str_2_list(a_list0)
b_list0 = str_2_list(b_list0)
c_list0 = str_2_list(c_list0)


index = [i for i in range(len(a_list0))]
np.random.shuffle(index)
for n in range(len(a_list0)):
    a_list.append(a_list0[index[n]])
    b_list.append(b_list0[index[n]])
    c_list.append(c_list0[index[n]])
    
 

print(a_list0)
print(b_list0)
print(c_list0)

print(a_list)
print(b_list)
print(c_list)