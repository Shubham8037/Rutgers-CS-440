# Binary Min Heap
import random

# This is the insert method for the binary heap. It will also
# make sure we are not visiting same neighbor twice
def insert(list, dict, fValue, cellNumber):
    if fValue in list:
        dict[fValue].append(cellNumber)
    else:
        dict[fValue] = [cellNumber]
        x = len(list)
        list.append(None)
        while x >0 and fValue < list[(x-1)//2]:
            list[x] = list[(x-1)//2]
            x = (x-1)//2
        list[x] = fValue

# This is a recursive method to minify the heap
def minify(list, x):
    size = len(list)
    left = x*2 +1
    right = x*2 + 2
    sorted = x
    
    if left < size -1 and list[left] < list[x]:
        sorted = left
    if right < size -1 and list[right] < list[sorted]:
        sorted = right
    if sorted != x:
        list[sorted], list[x] = list[x], list[sorted]
        minify(list,sorted)

def pop(list,dict):
    first = list[0]
    cellNumber = dict[first].pop(random.randrange(len(dict[first])))
    if len(dict[first]) == 0:
        del dict[first]
        list[0] = list[-1]
        list.pop()
        minify(list,0)
    return cellNumber
