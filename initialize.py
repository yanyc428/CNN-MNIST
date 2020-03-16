import requests
import os
import gzip

address1 = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
address2 = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
address3 = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
address4 = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

train_dir = './data/train'
test_dir = './data/test'

if not os.path.exists(train_dir):
    os.mkdir('./data')
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

adds = [address1, address2, address3, address4]
file_names = []
for add in adds:
    file_name = add.split('/')[-1].split('.')[0]
    if add is address1 or add is address2:
        file_name = os.path.join(train_dir, file_name)
    else:
        file_name = os.path.join(test_dir, file_name)
    file_names.append(file_name)


def un_gz(file_name):
    
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    

for i in range(4):
    response=requests.get(adds[i])
    with open(file_names[i], "wb") as code:
        code.write(response.content)
        un_gz(file_names[i])
        




