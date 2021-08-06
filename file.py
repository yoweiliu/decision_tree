import os
import time
 
time_start = time.time()

file_path = './data'
file_name = '/raw_data'


file_destination = os.getcwd() + file_path + file_name

if os.path.exists(file_destination):
    print ('yes')
else:
    print("' file_path '"+ file_destination + "' Not found'")


time_end = time.time()
print (time_end - time_start)

# print(os.getcwd())  # 相對位置
# print('abspath:',os.path.abspath(__file__))
