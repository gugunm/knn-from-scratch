import csv
import math
import operator

#ini untuk metode cross validationnya dibagi 2000 data train dan 2000 data test
def loadDataset(train1, train2, train3, test, trainingSet=[], testSet=[]):
    with open(train1, 'rb') as csvfile:
        linetrain1 = csv.reader(csvfile)
        dataset1 = list(linetrain1)
        for x in range(len(dataset1)):
            for y in range(4):
                dataset1[x][y] = float(dataset1[x][y])
            trainingSet.append(dataset1[x])
    with open(train2, 'rb') as csvfile:
        linetrain2 = csv.reader(csvfile)
        dataset2 = list(linetrain2)
        for x in range(len(dataset2)):
            for y in range(4):
                dataset2[x][y] = float(dataset2[x][y])
            trainingSet.append(dataset2[x])
    with open(train3, 'rb') as csvfile:
        linetrain3 = csv.reader(csvfile)
        dataset3 = list(linetrain3)
        for x in range(len(dataset3)):
            for y in range(4):
                dataset3[x][y] = float(dataset3[x][y])
            testSet.append(dataset3[x])
    with open(test, 'rb') as csvfile:
        linetest = csv.reader(csvfile)
        datatest = list(linetest)
        for i in range(len(datatest)):
            for j in range(4):
                datatest[i][j] = float(datatest[i][j])
            testSet.append(datatest[i])

#ini untuk testnya 1000, trainnya 4000
def loadDataset_and_datatest(train1, train2, train3, train4, test, trainingSet=[], testSet=[]):
    with open(train1, 'rb') as csvfile:
        linetrain1 = csv.reader(csvfile)
        dataset1 = list(linetrain1)
        for x in range(len(dataset1)):
            for y in range(4):
                dataset1[x][y] = float(dataset1[x][y])
            trainingSet.append(dataset1[x])
    with open(train2, 'rb') as csvfile:
        linetrain2 = csv.reader(csvfile)
        dataset2 = list(linetrain2)
        for x in range(len(dataset2)):
            for y in range(4):
                dataset2[x][y] = float(dataset2[x][y])
            trainingSet.append(dataset2[x])
    with open(train3, 'rb') as csvfile:
        linetrain3 = csv.reader(csvfile)
        dataset3 = list(linetrain3)
        for x in range(len(dataset3)):
            for y in range(4):
                dataset3[x][y] = float(dataset3[x][y])
            trainingSet.append(dataset3[x])
    with open(train4, 'rb') as csvfile:
        linetrain4 = csv.reader(csvfile)
        dataset4 = list(linetrain4)
        for x in range(len(dataset4)):
            for y in range(4):
                dataset4[x][y] = float(dataset4[x][y])
            trainingSet.append(dataset4[x])
    with open(test, 'rb') as csvfile:
        linetest = csv.reader(csvfile)
        datatest = list(linetest)
        for i in range(len(datatest)):
            for j in range(4):
                datatest[i][j] = float(datatest[i][j])
            testSet.append(datatest[i])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def getJumbenar(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return correct

#membaca data testing.csv
def bacaDataTest(namafile,test=[]):
    with open(namafile, 'rb') as file:
        read = csv.reader(file)
        datatest = list(read)
        for x in range(len(datatest)):
            for y in range(4):
                datatest[x][y] = float(datatest[x][y])
            test.append(datatest[x])

#menyimpan hasil prediksi dari data testing ke dalam file
def saveDataTest (fileinputname,fileoutputname, prediksi):
    with open(fileinputname, 'r') as input, open(fileoutputname, 'w') as output:
        reader = csv.reader(input, delimiter=',')
        writer = csv.writer(output, delimiter=',')
        all = []
        row = next(reader)
        all.append(row)
        it = prediksi.__iter__()
        for row in reader:
            if row:
                try:
                    row.append(next(it))
                except StopIteration:
                    row.append("")
                writer.writerow(row)

def maintest():
    trainingSet = []
    testSet = []
    predictions = []
    loadDataset_and_datatest('train1.csv', 'train2.csv', 'train3.csv', 'train4.csv', 'datatest.csv', trainingSet,testSet)
    print 'Train set: ' + repr(len(trainingSet)) + ' Test set: ' + repr(len(testSet))
    k = 51
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print '> predicted=' + repr(result) + ',index=' + repr((x)) #ini buat yang data test
    saveDataTest('datatest.csv', 'hasil_knn.csv', predictions) #nyimpen data test kedalam bentuk csv

def maintrain():
    trainingSet = []
    testSet = []
    predictions = []
    loadDataset('train1.csv', 'train2.csv', 'train3.csv', 'train4.csv', trainingSet, testSet) #ini untuk cross validationnya
    print 'Train set: ' + repr(len(trainingSet)) + ' Test set: ' + repr(len(testSet))
    k = 51
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print '> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]), ',index=' + repr(x) #Ini buat yang data train
    accuracy = getAccuracy(testSet, predictions) #buat liat akurasi data train
    jumbenar = getJumbenar(testSet, predictions) # buat liat jumlah benernya di proses training
    print('Accuracy: ' + repr(accuracy) + '%')
    print('JumBenar: ' + repr(jumbenar))

# MAIN PROGRAM.
maintrain()
maintest()