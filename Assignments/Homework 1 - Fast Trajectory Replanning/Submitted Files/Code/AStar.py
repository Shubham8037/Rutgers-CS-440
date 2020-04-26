import numpy as np
import random
from matplotlib.colors import ListedColormap
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import matplotlib 
import BinaryHeap as bh
import time 
import sys
from mazeGen import generateMazes

############## Part 1 
def isDeadEnd(y,x,visited) :     
	for i in range(-1,1):   #i in -1,0,1
		for j in range(-1,1):#i in -1,0,1
			if(not(i==0 and j==0)):#as if i==0 and j==0 then we are in teh same cell 	 
				if(x+i>=0 and x+i< num_cols ):
					if( y+j >=0 and y+j < num_rows):
						if((x+i,y+j) not in visited):
							return False,y+j,x+i #There's an unvisited neighbour 
	return True,-1,-1;                           #There's no unvisited neighbour
   
def isValidRow(y):
	if(y>=0 and y<num_rows):
		return True; 
	return False; 

def isValidCol(x):
	if(x>=0 and x<num_cols):
		return True; 
	return False; 
	
def generateMazes(mazesNumber,num_rows,num_cols):

	#Generate the 50 mazes 
	##########################################################
	# initially set all of the cells as unvisited
	maze      = np.zeros((mazesNumber,num_rows,num_cols))


	for mazeInd in range(0,mazesNumber) :
		print("Generate Maze : " + str(mazeInd+1));
		visited = set()  # Set for visitied nodes 
		stack   = []     # Stack is empty at first 

		##########################################################
		#start from a random cell	
		row_index = random.randint(0,num_rows-1)#Must choose valid row index 
		col_index = random.randint(0,num_cols-1)#Must choose valid col index 
		#mark it as visitied 	
		print("_______________ Start ________________\n")
		print("Loc["+str(row_index)+"],["+str(col_index)+"] = 1")
		visited.add((row_index , col_index))       #Visited 
		maze [mazeInd , row_index , col_index] = 1 #Unblocked 
		

		##########################################################
		#Select a random neighbouring cell to visit that has not yet been visited. 
		print("\n\n_______________ DFS ________________\n")
		while(len(visited) < num_cols*num_rows): #Repeat till visit all cells 
		
			crnt_row_index = row_index+random.randint(-1,1)#neighbor
			crnt_col_index = col_index+random.randint(-1,1)#neighbor
			i=0;isDead=False;
			while ((not isValidRow(crnt_row_index)) or (not isValidCol(crnt_col_index) )or ((crnt_row_index,crnt_col_index) in visited) ):
				# no need to write also "or (crnt_row_index==row_index and crnt_col_index==col_index)" as if this happened then it would be visited 
				crnt_row_index = row_index+random.randint(-1,1)
				crnt_col_index = col_index+random.randint(-1,1)
				i = i+1
				#print("dtuck"+str(i))
				if(i==8):
					#Reach dead end 
					isDead = True
					break
			if(not isDead):
				visited.add((crnt_row_index , crnt_col_index)) 
			
			rand_num  = random.uniform(0, 1)

			if( rand_num < 0.3 and not isDead) : 
				# With 30% probability mark it as blocked. 
				maze [mazeInd , crnt_row_index , crnt_col_index] = 0 #Leave the block  
				print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 0")				
				#to start get the neighbors of this cell next time 
				row_index = crnt_row_index
				col_index = crnt_col_index
			else : 
				if(not isDead):
					# With 70% mark it as unblocked and in this case add it to the stack.
					maze [mazeInd , crnt_row_index , crnt_col_index] = 1 #Unblocked 
					print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 1")				
					stack.append((crnt_row_index,crnt_col_index))
					isDead,unvisitRow , unvisitCol = isDeadEnd(row_index,col_index,visited)
				if(isDead == True):#if no unvisited neighbour 
					#backtrack to parent nodes on the search tree until it reaches a cell with an unvisited neighbour
					while(len(stack)>0):
						parent_row,parent_col = stack.pop();
						isDead,unvisitRow , unvisitCol = isDeadEnd(parent_row,parent_col,visited)
						if(isDead == False):
							break;
					# Now wither we reach not dead end or stack is empty 
					if(len(stack)>0):
						visited.add((unvisitRow,unvisitCol))
						row_index = unvisitRow
						col_index = unvisitCol
					else :
						#Repeat the whole process from a point not vistited
						row_index = random.randint(0,num_rows-1)
						col_index = random.randint(0,num_cols-1)
						if(len(visited)< num_cols*num_rows):
							while ( (not isValidRow(row_index)) or (not isValidCol(col_index)) or ((row_index,col_index) in visited) ):
								row_index = random.randint(0,num_rows-1)
								col_index = random.randint(0,num_cols-1)
								#print(str(row_index)+","+str(col_index))
						#mark it as visitied 	
						visited.add((row_index , col_index))       #Visited 					
				else : #No dead Node 
					visited.add((unvisitRow,unvisitCol))
					row_index = unvisitRow
					col_index = unvisitCol

				
	return maze
	
def poltline(A,col):
    x, y = zip(*A)
    temp, = plt.plot(y,x,color=col)
    print("Line type ")
    print(temp.__class__.__name__)
    return temp
	

# This method is to get Manhattan Distances for A*
def ManhattanDistanes(curr, goal):
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])

# This method will find the path for Forward A*
def getForwardPath(dict, curr, goal):
    temp = [goal]
    next = dict[goal]
    while next != curr:
        temp.insert(0,next)
        next = dict[next]
    temp.insert(0,next)
    return temp

# This method will find the path for the Backward A*
def getBackwardPath(dict, curr, goal):
    temp = [curr]
    next = dict[curr]
    while next != goal:
        temp.append(next)
        next = dict[next]
    temp.append(next)
    return temp

# This method will find the neighbors of the current cell
def nextNeighbor (cellNumber, length):
    neighborList = [(cellNumber[0],cellNumber[1]+1),(cellNumber[0],cellNumber[1]-1),(cellNumber[0]+1,cellNumber[1]),(cellNumber[0]-1,cellNumber[1])]
    temp = []
    for x in neighborList:
        if -1 < x[0] < length and -1 < x[0] < length:
            temp.append(x)
    return temp

# This method will update the neighbors to empty and blocked
def update(map,length,curr,bList):
    goodCells = nextNeighbor(curr,length)
    for x in goodCells:
        print(x)
        print(map[2][x])
		
        if x in bList:
            map[2][x] = 2 # It's blocked
        else:
            map[2][x] = 1 # It's empty


# This method will display the intial map

def dispMap(length, bList = None, pList = None, map = None, old = None):
    if not(map is None):
        d = map[2]
        f, a = plt.subplots()
        print(d)
        a.imshow(d)#, ListedColormap(['grey','white','black']))
    else:
        d = np.zeros((length, length))
        if not (bList is None):
            x = bList
        f, a = plt.subplots()
        a.imshow(d)#,  ListedColormap(['white','black']))
    print(a)
    #a.set_xticks(np.arange(( -0.5), length, 1))#, minor = True)
    #a.set_yticks(np.arange(( -0.5), length, 1))#, minor=True)
    a.grid(which= 'minor', color='black', linestyle='-', linewidth=1)
    
    if pList != None:
        poltline(pList, 'red')
    if old != None:
        poltline(old, 'green')
    plt.show()


def showLaunch(length, bList = None, pList = None, map = None, old = None): 
	#This method will add the colors to the maze 
    d = map[2]
    f,a = plt.subplots()
    colorMap = matplotlib.colors.ListedColormap(['grey','white','black'])
    boundaries = [-0.5,0.5,1.5,2.5]
    n = matplotlib.colors.BoundaryNorm(boundaries, colorMap.N)
    image = a.imshow(d,interpolation='nearest', origin='lower')
    a.set_xticks(np.arange(-.5, length, 1), minor=True)
    a.set_yticks(np.arange(-.5, length, 1), minor=True)
    a.grid(which= 'minor', color='black', linestyle='-', linewidth=1)
    oLine = poltline(old,'green')
    pLine = poltline(pList, 'red')
    plt.pause(1)
    return image, f, oLine, pLine

def showUpdate(image, f, oLine, pLine, map, aList, bList):
	# This method will add the path line to the maze 
    image.set_data(map)
    x,y = zip(*aList)
    oLine.set_xdata(y)
    oLine.set_ydata(x)
    x,y = zip(*bList)
    pLine.set_xdata(y)
    pLine.set_ydata(x)
    plt.pause(1)
    f.canvas.draw()

def adaptedAStar(mLength,bList, start, goal, show = False):
	# Here we implement Adaptive A* which will take the length, start, and goal as inputs
	maximumG = mLength * mLength
	currCell = start 
	gCell = goal
	magMap = np.zeros((5,mLength, mLength))
	magMap[3] = np.zeros((mLength,mLength), dtype = bool)
	magMap[4] = np.zeros((mLength,mLength), dtype = bool)
	magMap[2][currCell] = 1
	maxG = 0 
	totalSteps = 0 
	totalExpense = 0 
	currTrack = [start]
	update(magMap,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		magMap[1][currCell] = 0
		magMap[0][currCell] = totalSteps
		magMap[1][gCell] = np.inf
		magMap[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		magMap[4] = np.zeros((mLength,mLength), dtype = bool)
		bh.insert(oList, oDict, ManhattanDistanes(currCell,gCell), currCell)
		while oList and magMap[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			magMap[4][coloredCell] = True
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if magMap[2][newCell] != 2 :
					if magMap[3][newCell]:
						newHeuris = maxG - magMap[1][newCell]
					else:
						newHeuris = ManhattanDistanes(newCell,gCell)
					if magMap[0][newCell] < totalSteps:
						magMap[1][newCell] = np.inf
						magMap[0][newCell] = totalSteps
					if magMap[1][newCell] > magMap[1][coloredCell] + 1:
						fDict[newCell] = coloredCell
						newG = magMap[1][coloredCell] + 1
						bh.insert(oList, oDict, (maximumG*( newG +  newHeuris) -  newG) , newCell )
						magMap[1][newCell] = newG
		magMap[3] = magMap[4]
		maxG = magMap[1][gCell]
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = showLaunch(mLength, map = magMap, pList = currPath , old = currTrack)
				plt.show()
			else:
				showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if magMap[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update( magMap,mLength,currCell,bList)
				else:
					break 
	if show:
		showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
	return currTrack,totalExpense

def backwardAStarTie(mLength,bList, start, goal, show = False):
	# Here we implement Backward A* which will take the length, start, and goal as inputs
	totalExpense = 0 
	maxG = mLength*mLength
	currCell = start 
	gCell = goal
	magMap = np.zeros((4,mLength, mLength))
	magMap[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update(magMap,mLength,currCell,bList)
	while currCell != gCell:
		magMap[3] = np.zeros((mLength,mLength))
		totalSteps += 1 
		magMap[1][currCell] = np.inf
		magMap[0][currCell] = totalSteps
		magMap[1][gCell] = 0
		magMap[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.insert(oList, oDict, ManhattanDistanes(currCell,gCell), gCell)
		while oList and magMap[1][currCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			magMap[3][coloredCell] = 1
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if magMap[2][newCell] != 2 :
					if magMap[0][newCell] < totalSteps:
						magMap[1][newCell] = np.inf
						magMap[0][newCell] = totalSteps
					if magMap[1][newCell] > magMap[1][coloredCell] + 1:
						magMap[1][newCell] = magMap[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.insert(oList, oDict, maxG*(magMap[1][newCell] + ManhattanDistanes(newCell,currCell)) - magMap[1][newCell] , newCell )
		if not oList:
			return None, None
		
		if show:
			currPath = getBackwardPath(fDict,currCell,gCell)
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = showLaunch(mLength, map = magMap, pList = currPath , old = currTrack)
				plt.show()
			else:
				showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
		while currCell != gCell:
			cell = fDict[currCell]
			if magMap[2][cell] != 2 :
				currTrack.append(cell)
				currCell = cell
				update( magMap,mLength,currCell,bList)
			else:
				break
	if show:
		showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath) 
	return currTrack, totalExpense

def forwardAStarTie(mLength,bList, start, goal, show = False):
	# Here we implement Forward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	maxG = mLength* mLength
	currCell = start 
	gCell = goal
	magMap = np.zeros((4,mLength, mLength))
	magMap[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update(magMap,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		magMap[1][currCell] = 0
		magMap[0][currCell] = totalSteps
		magMap[1][gCell] = np.inf
		magMap[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.insert(oList, oDict, ManhattanDistanes(currCell,gCell), currCell)
		while oList and magMap[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			magMap[3][coloredCell] = 1
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if  magMap[2][newCell] != 2 :
					if magMap[0][newCell] < totalSteps:
						magMap[1][newCell] = np.inf
						magMap[0][newCell] = totalSteps
					if magMap[1][newCell] > magMap[1][coloredCell] + 1:
						magMap[1][newCell] = magMap[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.insert (oList, oDict, ( maxG*(magMap[1][newCell] + ManhattanDistanes(newCell,gCell)) - magMap[1][newCell] )  , newCell ) 
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = showLaunch(mLength, map = magMap, pList = currPath , old = currTrack)
				plt.show()
			else:
				showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if magMap[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update( magMap,mLength,currCell,bList)
				else:
					break 
	if show:
		showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
	return currTrack, totalExpense

def repeatedForwardAStar(mLength,bList, start, goal, show = False):
	# Here we implement Repeated Forward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	maxG = mLength* mLength
	currCell = start 
	gCell = goal
	magMap = np.zeros((4,mLength, mLength))
	magMap[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update(magMap,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		magMap[1][currCell] = 0
		magMap[0][currCell] = totalSteps
		magMap[1][gCell] = np.inf
		magMap[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.insert(oList, oDict, ManhattanDistanes(currCell,gCell), currCell)
		while oList and magMap[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if  magMap[2][newCell] != 2 :
					if magMap[0][newCell] < totalSteps:
						magMap[1][newCell] = np.inf
						magMap[0][newCell] = totalSteps
					if magMap[1][newCell] > magMap[1][coloredCell] + 1:
						magMap[1][newCell] = magMap[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.insert (oList, oDict, ( maxG*(magMap[1][newCell] + ManhattanDistanes(newCell,gCell)) + magMap[1][newCell] )  , newCell ) 
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = showLaunch(mLength, map = magMap, pList = currPath , old = currTrack)
				plt.show()
			else:
				showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if magMap[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update( magMap,mLength,currCell,bList)
				else:
					break 
	if show:
		showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
	return currTrack, totalExpense

def repeatedForwardAStarB(mLength,bList, start, goal, show = False):
	# Here we implement Repeated Forward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	currCell = start 
	gCell = goal
	magMap = np.zeros((3,mLength, mLength))
	magMap[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update(magMap,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		magMap[1][currCell] = 0
		magMap[0][currCell] = totalSteps
		magMap[1][gCell] = np.inf
		magMap[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.insert(oList, oDict, ManhattanDistanes(currCell,gCell), currCell)
		while len(oList)>0 and magMap[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			totalExpense +=1
			for newCell in nextNeighbor(coloredCell,mLength):
				if  magMap[2][newCell] != 2 :
					if magMap[0][newCell] < totalSteps:
						magMap[1][newCell] = np.inf
						magMap[0][newCell] = totalSteps
					if magMap[1][newCell] > magMap[1][coloredCell] + 1:
						magMap[1][newCell] = magMap[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.insert(oList, oDict, (magMap[1][newCell] + ManhattanDistanes(newCell,gCell)) , newCell )
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = showLaunch(mLength, map = magMap, pList = currPath , old = currTrack)
				plt.show()
			else:
				showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if magMap[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update( magMap,mLength,currCell,bList)
				else:
					break 
	if show:
		showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
	return currTrack,totalExpense

def repeatedBackwardAStar(mLength,bList, start, goal, show = False):
	# Here we implement Repeated Backward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	currCell = start 
	gCell = goal
	magMap = np.zeros((3,mLength, mLength))
	magMap[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update(magMap,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		magMap[1][currCell] = np.inf
		print("magMap")
		print(magMap [1,1,2])
		print("currCell")
		print( currCell )
		
		
		magMap[0][currCell] = totalSteps
		magMap[1][gCell] = 0
		magMap[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.insert(oList, oDict, ManhattanDistanes(currCell,gCell), gCell)
		print("olist")
		print(oList)
		while oList :
			print(magMap[1][currCell])
			
			if(magMap[1][currCell] <= oList[0]):
				break;
			totalExpense += 1
			coloredCell = bh.pop(oList, oDict)
			for newCell in nextNeighbor(coloredCell,mLength):
				if magMap[2][newCell] != 2 :
					if magMap[0][newCell] < totalSteps:
						magMap[1][newCell] = np.inf
						magMap[0][newCell] = totalSteps
					if magMap[1][newCell] > magMap[1][coloredCell] + 1:
						magMap[1][newCell] = magMap[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.insert(oList, oDict,magMap[1][newCell] + ManhattanDistanes(newCell,currCell), newCell )
		if not oList:
			return None, None
		
		if show:
			currPath = getBackwardPath(fDict,currCell,gCell)
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = showLaunch(mLength, map = magMap, pList = currPath , old = currTrack)
				plt.show()
			else:
				showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
		while currCell != gCell:
			cell = fDict[currCell]
			if magMap[2][cell] != 2 :
				currTrack.append(cell)
				currCell = cell
				update( magMap,mLength,currCell,bList)
			else:
				break 
	if show:
		showUpdate(im, fig, oLine,pLine, magMap[2],  currTrack ,currPath)
	return currTrack, totalExpense




if __name__ == '__main__':

	#####################################################
	# Generate mazes 101*101
	num_rows    = 101
	num_cols    = 101
	mazesNumber = 50
	#mazes = generateMazes(mazesNumber,num_rows,num_cols)
	#3D numpy array for the 50 mazes 
	

	#####################################################
	# Generate mazes 5*5
	num_rows    = 5
	num_cols    = 5
	mazesNumber = 3
	mazes2 = generateMazes(mazesNumber,num_rows,num_cols)

	mLength = 5;
	bList = [(2,3),(3,4),(3,3),(4,3),(4,4),(5,4)];
	start = (1,1);
	goal  = (3,2);
	
	#np.savetxt('maze '+str(mazeInd)+'.txt',mazes[mazeInd].astype(int) ,fmt='%i', delimiter=",");

	length= 5            ;
	bList = [(2,3),(3,4),(3,3),(4,3),(4,4),(5,4)]; # Example given in assignment
	pList = [(3,1),(3,4)];
	map   = mazes2     ;
	old   = None         ;
	dispMap(list, bList  , pList  , map  , old  );
	
	startX =input("Please , Enter valid x coordinate for the start point");
	startY =input("Please , Enter valid y coordinate for the start point");
	
	goalX  =input("Please , Enter valid x coordinate for the goal point"); 
	goalY  =input("Please , Enter valid y coordinate for the goal point"); 
	

	#####################################################
	# Repeated Forward A*
	print("Repeated Forward A*")
	repeatedForwardAStarB(mLength,bList, start, goal, True);
	print("____________________________________")

	#####################################################
	# Repeated Backward A*
	print("Repeated Backward A*")
	repeatedBackwardAStar(mLength,bList, start, goal, True);
	print("____________________________________")

	#####################################################
	# Repeated forward A with smaller g*
	print("Repeated Forward A* with smaller g-values")
	repeatedForwardAStar(mLength,bList, start, goal, True);
	print("____________________________________")


	#####################################################
	# Adaptive A*
	print("Adapted A*")
	adaptedAStar(mLength,bList, start, goal, True);
		
	#####################################################
	# forward A large g* 

	print("Repeated Forward A* with large g-values")
	forwardAStarTie(mLength,bList, start, goal, True);
	print("____________________________________")

	#####################################################
	# backward A large g* 

	print("Repeated Backward A* with large g-values")		
	backwardAStarTie(mLength,bList, start, goal, True);
	print("____________________________________")
	
	