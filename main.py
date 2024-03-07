import random
import csv
import copy
import sys


def recursive_function(term1, term2):
    #Separate each term into 2 a,b,c,d
    term1 = str(int(term1))
    term2 = str(int(term2))
    n = len(term1)//2
    a = term1[:len(term1)-n]
    b = term1[len(term1)-n:]
    c = term2[:len(term2)-n]
    d = term2[len(term2)-n:]

    #Step 1: Multiply a by c
    if len(a) > 2 or len(c) > 2:
        ac = recursive_function(a,c)
    else:
        ac = int(a) * int(c)
    #Step 2: Multiply b by d
    if len(b) > 2 or len(d) > 2:
        bd = recursive_function(b,d)
    else:
        bd = int(b) * int(d)
    #Step 3: compute (a + b)(c+d)
    ab = str(int(a) + int(b))
    cd = str(int(c) + int(d))
    if len(ab) > 2 or len(cd) > 2:
        abcd = recursive_function(ab,cd)
    else:
        abcd = int(ab) * int(cd)
    abcd = int(abcd - ac - bd)
    x = (10**n)**2*ac + 10**(n)*abcd + bd
    return int(x)

    #product of term1 *  term2

#print(recursive_function(3141592653589793238462643383279502884197169399375105820974944592,2718281828459045235360287471352662497757247093699959574966967627))

def IntegerArray():
    file = open("IntegerArray.txt")
    A = [0]*100000
    #A = [6,5,4,3,2,1]
    for i, line in enumerate(file.readlines()):
        A[i] = int(line)
    A = merge_sort(A,0)
    print(A)

def merge_sort(A,inversions):
    # Merge Sort
    n = len(A)
    if n == 1:
        return (A,0)
        # sort the left half
    D = [0]*n
    B = merge_sort(A[:n//2],0)
    inversions += B[1]
    B = B[0]
        # sort the right half
    C = merge_sort(A[n//2:],0)
    inversions += C[1]
    C = C[0]
    # merge
    i = j = 0

    for k in range(n):
        if i == len(B):
            D[k] = C[j]
            j += 1
        elif j == len(C):
            D[k] = B[i]
            i += 1
        elif B[i] < C[j]:
            D[k] = B[i]
            i += 1
        elif B[i] > C[j]:
            inversions += (len(B[i:]))
            D[k] = C[j]
            j += 1
    return D, inversions

def quick_sort(A,comparisons):
    #select first value in array as pivot

    if len(A) <= 1:
        return A,0
    else:
        comparisons = len(A) - 1
    j = 1
    #to change pivot point, swap pivot with first index of array

    #Select pivot point based on 1st middle last
    if len(A)%2 == 0:
        mid = A[len(A)//2-1]
        mid_index = len(A)//2 - 1
    else:
        mid = A[len(A)//2]
        mid_index = len(A)//2
    array = [A[0],mid,A[-1]]
    array_index = [0, mid_index, -1]
    array = quick_sort_median(array,array_index)
    #sort array
    #array[1][1] = -1

    pivot = A[array[1][1]]
    A[array[1][1]] = A[0]
    A[0] = pivot
    for i in range(1,len(A)):
        if A[i] < A[0]:
            #place to left of A[0] and do swap
            x = A[j]
            A[j] = A[i]
            A[i] = x
            j += 1
    x = A[0]
    A[0] = A[j-1]
    A[j-1] = x

    #sort right half
    B = quick_sort(A[j:],0)
    comparisons += B[1]
    #sort left half
    C = quick_sort(A[:j-1],0)
    comparisons += C[1]
    return C[0]+[A[j-1]]+B[0], comparisons


def quick_sort_median(A,A_int):
    # select first value in array as pivot

    if len(A) <= 1:
        return A, A_int
    j = 1
    # to change pivot point, swap pivot with first index of array
    for i in range(1, len(A)):
        if A[i] < A[0]:
            # place to left of A[0] and do swap
            x = A[j]
            y = A_int[j]
            A[j] = A[i]
            A_int[j] = A_int[i]
            A[i] = x
            A_int[i] = y
            j += 1
    # sort right half
    B = quick_sort_median(A[j:], A_int[j:])

    # sort left half
    C = quick_sort_median(A[1:j], A_int[1:j])

    return C[0] + [A[0]] + B[0], C[1] + [A_int[0]] + B[1]



def contraction(foo):
#pick random edge
    start = random.choice(list(foo.keys()))
    #start = 1
    end = foo[start][0]
    if len(foo) <=  2:
        edges = len(foo[start])             #end = 48
        return edges #contract it
    #This means find and replace all refrences of one and replace with another. Except for if it occurs on itself.

    foo[end] = foo[end] + foo[start]
    for i in range(len(foo[end])):
        if foo[end][i] == start:
            foo[end][i] = -1
    for i in foo[start]:
        for j in range(len(foo[i])):
            if foo[i][j] == start:
                foo[i][j] = end

    foo[end] = [j for j in foo[end] if int(j) > 0 and j != end]
    foo.pop(start)
    edges = contraction(foo)
    return edges
    #delete any self loops

def binary_search(A, target):
    #pick middle
    if A[len(A)//2][0] == target:
        #walk backwards until finding first instance
        x = A[len(A)//2][0]
        count = -1
        if A[0][0] == target:
            return 0
        while x == target and len(A)//2 - count > 0:
            count += 1
            x = A[len(A)//2 - count][0]
        return len(A)//2 - count +1
    elif A[len(A)//2][0] < target:
        #look in second half
        if len(A) == 1:
            return "Not found"
        return binary_search(A[len(A)//2:],target) + len(A)//2

    else:
        if len(A) == 1:
            return "Not found"
        return binary_search(A[:len(A)//2],target)
import time
def DFS(A,start_node,explored,n):
    current_node = start_node
    stack = [current_node]
    if explored[current_node] != 0:
        return explored, n
    else:
        explored[current_node] = -1
    # Move to next node
    i = 0
    j = -1
    while stack:
        try:
            placeholder_node = A[current_node][i]
            # Add to stack
            if explored[placeholder_node] == 0:
                stack += [placeholder_node]
                current_node = placeholder_node
                explored[current_node] = -1
                i = 0
            else:
                i += 1
            # repeat until can't find next node
        except:
            #Retreat backwards mark order
            n += 1
            explored[current_node] = n

            stack = stack[:-1]
            i = 0
            if len(stack) > 0:
                current_node = stack[j]
    # Retreat backwards a step
    return explored, n





    #If there is nowhere to go, mark count and increment count then  return
# How to mark as read?


import csv
import copy
def read_inputs():
    A = {}
    B = {}
    count = set()
    with open('Strongest_connections.txt', 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            if int(row[0]) in A:
                A[int(row[0])] += [int(row[1])]
            else:
                A[int(row[0])] = [int(row[1])]
            if int(row[1]) in B:
                B[int(row[1])] += [int(row[0])]
            else:
                B[int(row[1])] = [int(row[0])]
            count.add(row[0])
            count.add(row[1])
    return A, B, len(count)




#A = [[1,2],[1,4],[2,3],[4,3]]
#A = {1:[4],2:[8], 3:[6], 4:[7], 5:[2],6:[9],7:[1],8:[5,6],9:[3,7]}
#B = {4:[1],8:[2],6:[3,8],7:[4,9],2:[5],9:[6],1:[7],5:[8],3:[9]}

def strong_components():
    A, B, count = read_inputs()
    explored = [0] * 875715
    explored = {i: 0 for i in range(1, count + 1)}
    num_nodes = 0
    for i in range(1,const_nodes+1):
        explored, num_nodes = DFS(B,i,explored, num_nodes) #eplored{node:n}
    explored = {v: k for k, v in explored.items()}#explored{n:node}
    print(explored)
    final = [0]*(const_nodes+1)
    num_nodes = 0
    x = 0
    output = []
    for i in range(const_nodes, 0, -1):
        final, num_nodes = DFS(A,explored[i],final, num_nodes)
        output += [num_nodes - x]
        x = num_nodes
        print(num_nodes)
    output = sorted(output)
    print(output)


def dijkstra_read_inputs():
    G = {}
    A = {}
    with open("dijkstra.txt") as f:
        for row in csv.reader(f,delimiter='\t'):
            G[int(row[0])] = [[int(i) for i in row[1].split(",")]]
            for i in range(2,len(row)):
                if row[i] != '':
                    G[int(row[0])] += [[int(j) for j in row[i].split(",")]]
            #G[int(row[0])] = [int(G[int(row[0])][i]) for i in range(len(G[int(row[0])]))]
    return G
def dijkstra(s,t,num_nodes):

    G = dijkstra_read_inputs()
    #G = {1:[[2,1],[4,3]],2:[[3,3],[1,1],[4,1],[5,1]],4:[[3,2],[1,3],[2,1]],3:[[2,3],[4,2],[5,1]],5:[[2,1],[3,1]]}
    A = {s:0}
    B = {s:str(s)}
    #find shortest path from s to t
    #similar to breadth first search
    #Need explored? or A
    greedy_score = 10000000
    end_node = 0
    start_node = 0
    while t not in A:
        for i in A:
            if i == 112:
                pass
            if i == 26:
                pass
            #check greedy scores of all i paths
            for j in G[i]:
                if j[1] + A[i] < greedy_score and j[0] not in A: #Add to A
                    greedy_score = j[1] + A[i]
                    end_node = j[0]
                    start_node = i
        if end_node == 18:
            pass
        A[end_node] = greedy_score
        B[end_node] = B[start_node] +','+ str(end_node)
        #G[start_node] = [x for x in G[start_node] if x[0] != end_node]
        greedy_score = 10000000
    return A
    #use Dijkstra greedy score to compute which vertex to add next

def heap_insert(heap, value, value_index,min_heap):
    #append to end of heap
    heap.append(value)
    #Maintain heap property
    if min_heap == True:
        while heap[(value_index)//2] > value:
            #Swap values
            x = heap[(value_index)//2]
            heap[(value_index) // 2] = value
            heap[value_index] = x
            value_index = (value_index)//2
    else:
        while heap[(value_index)//2] < value:
            #Swap values
            x = heap[(value_index)//2]
            heap[(value_index) // 2] = value
            heap[value_index] = x
            value_index = (value_index)//2
            if value_index == 1:
                break
    return heap

def heap_delete_min_max(heap, min_heap):
    #swap first and last + delete former root
    heap[1] = heap[-1]
    heap.pop()
    #Now bubble down
    root_index = 1
    if min_heap == True:
        try:
            while heap[root_index] > heap[root_index*2] or heap[root_index] > heap[root_index*2+1]:
                #swap with min of the two
                if heap[root_index*2] < heap[root_index*2+1]:
                    x = heap[root_index*2]
                    heap[root_index*2] = heap[root_index]
                    heap[root_index] = x
                    root_index = 2*root_index
                else:
                    x = heap[root_index * 2+1]
                    heap[root_index*2+1] = heap[root_index]
                    heap[root_index] = x
                    root_index = 2 * root_index +1
        except:
            if len(heap)%2 == 1:
                #swap root index with final position
                x = heap[root_index]
                heap[root_index] = heap[-1]
                heap[-1] = x
    else:
        try:
            while heap[root_index] < heap[root_index*2] or heap[root_index] < heap[root_index*2+1]:
                #swap with max of the two
                if heap[root_index*2] < heap[root_index*2+1]:
                    x = heap[root_index*2+1]
                    heap[root_index*2 + 1] = heap[root_index]
                    heap[root_index] = x
                    root_index = 2*root_index +1
                else:
                    x = heap[root_index * 2]
                    heap[root_index * 2] = heap[root_index]
                    heap[root_index] = x
                    root_index = 2 * root_index
        except:
            if len(heap)%2 == 1:
                #swap root index with final position
                x = heap[root_index]
                heap[root_index] = heap[-1]
                heap[-1] = x
    return heap

def rebalance_heap(heap_max,heap_min, count_max, count_min):
    while count_max - count_min > 1 or count_max - count_min < 0:
        if count_max > count_min:
            #delete heap[0] from heap_max and insert it into heap_min

            heap_min = heap_insert(heap_min, heap_max[1], count_min, True)
            heap_max = heap_delete_min_max(heap_max, False)
            count_min += 1
            count_max -= 1
        else:

            heap_max = heap_insert(heap_max, heap_min[1], count_max, False)
            heap_min = heap_delete_min_max(heap_min, True)
            count_max += 1
            count_min -= 1
    return heap_max, heap_min, count_max, count_min
def median_maintain(A):
    #Create two heaps. One for mins and one for max's
    if A[0] > A[1]:
        heap_max = [0,A[1]]
        heap_min = [0,A[0]]
        sum_median = A[0] + A[1]
    else:
        heap_max = [0,A[0]]
        heap_min = [0,A[1]]
        sum_median = A[0] + A[0]
    count_max = 2
    count_min = 2
    for value in A[2:]:
        #determine which heap to add to.
        if value < heap_max[1]:
            heap_max = heap_insert(heap_max, value, count_max, False)
            count_max += 1
        elif value > heap_min[1]:
            heap_min = heap_insert(heap_min, value, count_min, True)
            count_min += 1
        else:
            heap_max = heap_insert(heap_max, value, count_max, False)
            count_max += 1
        #Maintain equal split by rebalancing
        heap_max,heap_min,count_max,count_min = rebalance_heap(heap_max,heap_min,count_max, count_min)
        sum_median += heap_max[1]
    return sum_median
import time
start_time = time.time()

with open("median_maintain.txt") as f:
    A = [int(line.rstrip()) for line in f]
#A = [6,5,4,3,2,1]
#A = [1,666,10,667,100,2,3]
#A = [1,2,3,4,5,6]
#A = [6331,2793,1640,9290,225,625,6195,2303,5685,1354]
#A = [3,2,4,1,5]
print(median_maintain(A)%10000)

print("--- %s seconds ---" % (time.time() - start_time))