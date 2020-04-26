import numpy as np

def featuresExtract(datum,type):
    if type==0:
        return basicFeaturesExtract(datum)
    else:
        return advancedFeaturesExtract(datum)

def batchExtract(data,type):
    if type==0:
        return basicBatchExtract(data)
    else:
        return advancedBatchExtract(data)

def basicFeaturesExtract(datum):
    features = []
    for x in range(datum.height):
        row = []
        for y in range(datum.width):
            if datum.getPixel(x, y) > 0:
                row.append(1)
            else:
                row.append(0)
        features.append(row)
    return np.array(features)

def isValid(s,w,h):
    #distinguish whether out of boundary
    x,y=s
    return (x>=0 and y>=0 and x<h and y<w)

def getNeighbor(s,w,h,visited):
    x,y=s
    neighbors=[]
    if isValid((x-1,y),w,h) and not visited[x-1][y]:
        neighbors.append((x-1,y))
    if isValid((x+1,y),w,h) and not visited[x+1][y]:
        neighbors.append((x+1,y))
    if isValid((x,y-1),w,h) and not visited[x][y-1]:
        neighbors.append((x,y-1))
    if isValid((x,y+1),w,h) and not visited[x][y+1]:
        neighbors.append((x,y+1))
    return neighbors

def advancedFeaturesExtract(datum):
    #find cycles in the image using DFS
    width=datum.width
    height=datum.height
    #cycle cnt
    cycle=-1
    features = basicFeaturesExtract(datum)
    #visited->1;not->0
    visited=np.empty_like((height,width))
    visited=np.array(features)
    while not (np.count_nonzero(visited)==visited.size):
        open=[]
        #get next 0
        open.append(np.unravel_index(visited.argmin(),visited.shape))
        while not (len(open)==0) :
            current=open.pop()
            visited[current[0]][current[1]]=1
            neighbors=getNeighbor(current,width,height,visited)
            for nb in neighbors:
                if not features[nb[0]][nb[1]]:
                    open.append(nb)
        cycle+=1
    cyclef=np.zeros((1,width))
    for i in range(cycle):
        cyclef[0][i]=1
    features=np.vstack((features,cyclef))
    return features


def basicBatchExtract(data):
    features = np.empty((len(data), data[0].width * data[0].height))
    for i in range(len(data)):
        feature = np.array(basicFeaturesExtract(data[i])).flatten()
        features[i, :] = feature
    return features.transpose()


def advancedBatchExtract(data):
    features = np.empty((len(data), data[0].width * (data[0].height+1)))
    for i in range(len(data)):
        feature = np.array(advancedFeaturesExtract(data[i])).flatten()
        features[i, :] = feature
    return features.transpose()
