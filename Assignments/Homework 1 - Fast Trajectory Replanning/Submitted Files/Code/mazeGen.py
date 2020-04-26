import numpy as np 
import random 

#Build the maze using depth first search 
num_rows    = 5
num_cols    = 5
mazesNumber = 50


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
	
def generateMazes():

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
		
if __name__ == '__main__':
	mazes = generateMazes() #3D numpy array for the 50 mazes 
	
	for mazeInd in range(0,mazesNumber):
		#np.savetxt(f, result.astype(int),, delimiter=",")
		 
		np.savetxt('maze '+str(mazeInd)+'.txt',mazes[mazeInd].astype(int) ,fmt='%i', delimiter=",")