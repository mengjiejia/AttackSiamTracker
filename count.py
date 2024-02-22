import os

# folder path
dir_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/training_dataset/got10k/crop511'
count = 0
# Iterate directory
count = 0
for path in os.listdir(dir_path):
    # check if current path is a file
    folder = os.path.join(dir_path, path)
    for v in os.listdir(folder):
        v_path = os.path.join(folder, v)
        count = count + len(os.listdir(v_path))
print(count)
print('File count:', count / 2)
a = 'abc'
if a:
    print('no problem')

print(100//2)
