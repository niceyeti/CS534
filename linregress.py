from __future__ import print_function
import numpy as np
from sklearn import linear_model
import sys
import matplotlib.pyplot as plt

"""
I need to memorize this: how to prepend a column of ones (for model bias input) 
to a np matrix.

@npMatrix: A numpy matrix
"""
def PrependBiasCol(npMatrix):
	return np.insert(npMatrix, 0, 1.0, axis=1)


"""
Returns x/y sequences as a tuple generated from these coefficients. This is only good for 2d data.

@xs: A sequence of x values; bin by these and use as inputs to generate a y value for each bin
@coefs: the regression coefficients. coef[0] is treated as the x0 coefficient, coef[1] as the x1/y coefficient.

Returns: (xs,ys), a tuple of x and corresponding dependent/y outputs, for plotting a line
"""
def GenerateDiscriminantSeq1(xs, coefs, intercept):
	maxX = max(xs)
	minX = min(xs)
	bins = 5 #just a few is sufficient to plot a line
	binWidth = (maxX - minX) / bins
	#generate the sample points of the least squares-fit line
	xsamples = [(minX + binWidth*float(i)) for i in range(0,bins)]
	ysamples = [(x * (-coefs[0] / coefs[1]) - intercept / coefs[1]) for x in xsamples]
	##plt.plot(xsamples, ysamples)
	#don't show; may be more stuff to plot
	return (xsamples,ysamples)


"""
Generates an x an y sequence for plotting, given some fitted linear regression model.

@lr: an sklearn fitted linear regression model
@data: ndarray of x values
"""
def GenerateDiscriminantSeq(lr, xdata):
	xs = [x[0] for x in xdata]
	minX = 10000
	minIndex = -1
	maxX = -10000
	maxIndex = -1
	for i in range(0,len(xs)):
		if xs[i] < minX:
			minX = xs[i]
			minIndex = i
		if xs[i] > maxX:
			maxX = xs[i]
			maxIndex = i

	ys = lr.predict(xdata)
	xs = [xs[minIndex],xs[maxIndex]]
	ys = [ys[minIndex],ys[maxIndex]]
	
	return xs, ys


"""
Given a fitted linear-model object, an X matrix of inputs and y vector of 
intended outputs, calculates the confusion matrix. Also reports the ssr's and #misclassified.

Treat "5" class as the 'positive' class, which is consistent with the input data.

Xs: Matrix of dimension NxM
ys: vector of labels of dimension N
lr: A linear_model.LinearRegression object, already fitted
"""
def calculateConfusionMatrix(Xs,ys,lr):
	misclassifiedCt = 0
	fpCt = 0
	tpCt = 0
	fnCt = 0
	tnCt = 0

	i = 0
	while i < len(Xs):
		x = Xs[i]
		y = int(ys[i])
		predictedVal = lr.predict([x])
		#clamp the prediction to the sign of the predicted value
		if predictedVal > 0:
			predictedLabel = 1
		else:
			predictedLabel = -1
		
		if predictedLabel == 1 and y == 1:
			tpCt += 1
		elif predictedLabel == -1 and y == 1:
			fnCt += 1
			misclassifiedCt += 1
		elif predictedLabel == -1 and y == -1:
			tnCt += 1
		elif predictedLabel == 1 and y == -1:
			fpCt += 1
			misclassifiedCt += 1
		else:
			print("ERROR no case found ???")		
		i += 1

	#output the result stats	
	misclassifiedRate = 100 * float(misclassifiedCt) / float(len(Xs))
	print("SSR: "+str(lr.residues_))
	print("#Misclassified: "+str(misclassifiedCt))
	print("Misclassified rate: "+str(misclassifiedRate)+"%")
	print("\nConfusion Matrix:")
	print("   +      -      row total")
	print("+  "+str(tpCt)+"   "+str(fpCt)+"   "+str(tpCt+fpCt))
	print("-  "+str(fnCt)+"   "+str(tnCt)+"   "+str(fnCt+tnCt))


"""
Scatter plots some 2d data.
@data: A np matrix

def scatterplotData(data, markerStyle):
	fig, ax = plt.subplots()
	ax.scatter(data[:,0], data[:,1], marker=markerStyle)
"""

def usage():
	print("Usage: python ./linregress.py [input csv file]")

if len(sys.argv) < 2:
	print("Wrong number of params")
	usage()
	exit()

dataset = np.loadtxt(sys.argv[1], delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:2]
y = dataset[:,2]
#add a bias col to the data
#X = PrependBiasCol(X)

lr = linear_model.LinearRegression()
#print(str(X))
lr.fit(X,y)
print("coefficients: "+str(lr.coef_)+"  y-intercept: "+str(lr.intercept_)+"  residues: "+str(lr.residues_))
#print(str(x0s))

#get the x0 values, the 0th component of all the x vectors.
x0s = list(dataset[:,0])
xs, ys = GenerateDiscriminantSeq1(x0s, lr.coef_, lr.intercept_)
#xs, ys = GenerateDiscriminantSeq(lr,X)
fig, ax = plt.subplots()
#partition the data for scatter plotting
positiveData = np.array([row for row in dataset if row[2] > 0])
negativeData = np.array([row for row in dataset if row[2] < 0])
#scatterplot the + examples
ax.scatter(positiveData[:,0], positiveData[:,1], marker="o")
#scatterplot the - examples
ax.scatter(negativeData[:,0], negativeData[:,1], marker="v")
#plot the discriminant line
ax.plot(xs, ys)

#some manual verification
testy = 0.1 * -lr.coef_[0] / lr.coef_[1] - lr.intercept_ / lr.coef_[1]
print(str(testy))
label = 0.1 * lr.coef_[0] + -2 * lr.coef_[1] + lr.intercept_
print("label: "+str(label)+"  predict: "+str(lr.predict([[0.1,-2.0]])))

#show
plt.show()

calculateConfusionMatrix(X,y,lr)





