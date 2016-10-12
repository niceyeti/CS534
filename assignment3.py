

import random
import numpy as np
import matplotlib.pyplot as plt
import math


def f(x):
    return 2 * math.sin(1.5* x)

"""
Given a 2d dataset of (x,y) datapoints, calculates the linear solution to (X*X)^-1*X*y,
using linear regression.

@xseq: A sequece of x input values
@yseq: The output/detpendent values given by the x values in xseq.

Return: The coefficients of the polynomial regression solution, as well as the residuals for the solution.
"""
def MyPolyFit(xseq, yseq, deg):
    #start by building a matrix of size |xseq|*deg
    #transform the data x values into non-linear transforms
    zdata = []
    for x in xseq:
        row = []
        for i in range(0,deg+1):
            row.append(x**float(i))
        zdata.append(row)
    print("zdata: "+str(zdata))
    M = np.matrix(zdata)
    #get the pseudo inverse of M
    print("shape: "+str(M.shape))
    print(str(M))
    M_inv = np.linalg.pinv(M)
    print("inv shape: "+str(M_inv.shape))
    print("yseq len: "+str(len(yseq)))
    print(str(M_inv))
    #now get the coefficients by simply multiplying the pseudo-inverse by the solution vector
    coefs = M_inv * np.transpose(np.matrix(yseq))
    #now get the residuals for the solution
    y_preds = [sum([coefs[i]*x**i for i in range(0,deg+1)]) for x in xseq]
    ssr = GetSSR(y_preds, yseq)
    
    return coefs, ssr
    
"""
Generates noiseless data; this is just for plotting.

returns: xseq, yseq
"""
def GenerateTrueDataset(n):
    xseq = []
    yseq = []
    step = 5.0 /float(n)    
    
    for i in range(0,n):
        x = i * step
        y = f(x)
        xseq.append(x)
        yseq.append(y)
    
    return xseq, yseq
    
    
"""
Generates data from the true target function, f(x) = 2sin(1.5x)+N(0,1), where N(0,1) is a Gaussian of zero mean, 1 variance.
X vaues are constrained to lie in 0 to 5.

@n: size of the dataset
"""
def GenerateDataset(n):
    xseq = []
    yseq = []
    step = 5.0 / float(n)
    
    for i in range(0,n):
        x = i * step
        y = f(x) + random.gauss(0,1.0)
        xseq.append(x)
        yseq.append(y)
    
    return xseq, yseq
    
"""
Given some polynomial coefficients, generates a bunch of data points based on this
polynomial in range [0,5].

@coef: The polynomial coefficients, largest first (np format). The degree of the polynomial is inferred from the length of this sequence.
@xin: The inputs for which to compute g(x)

Returns: xseq, yseq of datapoints corresponding for the input xin indices
"""
def GeneratePolyData(coef, xin):
    yseq = []
    xseq = []
    
    #reverse the coefficients, for readability
    coefs = [coef[len(coef) - i - 1] for i in range(0, len(coef))]
    
    for x in xin:
        xseq.append(x)
        #ugly coef indexing is so the coefficients are accessed in the reverse order, per np.polyfit order
        components = [coefs[i] * x**float(i) for i in range(0,len(coef))]
        y = sum(components)
        yseq.append(y)

    return xseq, yseq

"""
Calculates the ssr values of two different pairwise sequences of y-values; NOTE
this assumes the y values were indexed/generated with the same x values.
"""
def GetSSR(y1Vals, y2Vals):
    if len(y1Vals) != len(y2Vals):
        print("ERROR len y1Vals="+str(len(y1Vals))+" != y2Vals="+str(len(y2Vals))+"  in GetSSR()")

    ssr = 0.0
    for i in range(0,len(y2Vals)):
        ssr += ((y1Vals[i] - y2Vals[i])**2)
    
    return ssr    
    
"""
Dataset: A tuple, the first member being an xsequence, the second member a dependent y sequence.
The dataset (x,y) pairs are sorted by x values, and returned again as a tuple like the input.
"""
def SortDataByXVal(dataset):
    tupList = [(dataset[0][i], dataset[1][i]) for i in range(0,len(dataset[0]))]
    temp = sorted(tupList, key=lambda x: x[0])
    sortedData = [datum[0] for datum in temp], [datum[1] for datum in temp]
    return sortedData
    

"""
@xseq a list of x values
@yseq: dependent y values
@style: string like "b--" (blue, dashed line). See matplot lib for other shorthand params.
"""
def ScatterplotDataset(xseq, yseq,style):
    plt.scatter(xseq, yseq, marker=style)

ntrain = 25
nval = 75
#generate a dataset
trueX, trueY = GenerateTrueDataset(ntrain+nval)
trainData = GenerateDataset(ntrain)
valData = GenerateDataset(nval)
completeData = trainData[0]+valData[0], trainData[1]+valData[1]
completeData = SortDataByXVal(completeData)
#print("Complete data: "+str(completeData))

#plot the true function
plt.plot(trueX, trueY, "g")
#plot the training data, the noisy outputs of the true function
ScatterplotDataset(trainData[0],trainData[1],"*")
plt.show()
plt.clf()

#fit the training data using polynomial regression, with various degree parameters, and plot
coefs1, res1, rank, sv1, rcond  = np.polyfit(trainData[0], trainData[1], 1, full = True)
coefs2, res2, rank, sv2, rcond = np.polyfit(trainData[0], trainData[1], 2, full = True)
coefs3, res3, rank, sv3, rcond = np.polyfit(trainData[0], trainData[1], 3, full = True)
coefs4, res4, rank, sv4, rcond = np.polyfit(trainData[0], trainData[1], 4, full = True)
coefs5, res5, rank, sv5, rcond = np.polyfit(trainData[0], trainData[1], 5, full = True)

coefs2, res2 = MyPolyFit(trainData[0], trainData[1], 2)
print("my coefs: "+str(coefs2)+"    "+str(res2))
coefs2, res2, rank, sv2, rcond = np.polyfit(trainData[0], trainData[1], 2, full = True)
print("np coefs: "+str(coefs2)+"    "+str(res2))

print("np.polyfit residuals: "+str(res1)+" "+str(res2)+" "+str(res3)+" "+str(res4)+" "+str(res5))
#print("coefs5: "+str(coefs5))
#generate data from each model estimates
G1 = GeneratePolyData(coefs1, trainData[0])
#print("xseqs:\n"+str(G1[0])+"\n"+str(trainData[0]))
G2 = GeneratePolyData(coefs2, trainData[0])
G3 = GeneratePolyData(coefs3, trainData[0])
G4 = GeneratePolyData(coefs4, trainData[0])
G5 = GeneratePolyData(coefs5, trainData[0])

#get the ssr's per model
ssr1_train = GetSSR(G1[1], trainData[1]) #get the Ein value for this polynomial
ssr2_train = GetSSR(G2[1], trainData[1]) #get the Ein value for this polynomial
ssr3_train = GetSSR(G3[1], trainData[1]) #get the Ein value for this polynomial
ssr4_train = GetSSR(G4[1], trainData[1]) #get the Ein value for this polynomial
ssr5_train = GetSSR(G5[1], trainData[1]) #get the Ein value for this polynomial


#plot all of the g() estimates
plt.plot(G1[0],G1[1],"b-")
plt.plot(G2[0],G2[1],"g-")
plt.plot(G3[0],G3[1],"y-")
plt.plot(G4[0],G4[1],"r-")
plt.plot(G5[0],G5[1],"rs")
plt.show()
plt.savefig("AllHypotheses.png")
plt.clf()

#generate validation outputs from the training-derived parameters, and measure their Eval
#NOTE: this is messy coding. These calls assume that the generate polynomial data is x-aligned with the valData generated previously.
GV1 = GeneratePolyData(coefs1, valData[0])
GV2 = GeneratePolyData(coefs2, valData[0])
GV3 = GeneratePolyData(coefs3, valData[0])
GV4 = GeneratePolyData(coefs4, valData[0])
GV5 = GeneratePolyData(coefs5, valData[0])

#print("xseqs:\n"+str(GV1[0])+"\n"+str(valData[0]))

#get the ssr values per model
ssr1_val = GetSSR(GV1[1], valData[1])
ssr2_val = GetSSR(GV2[1], valData[1])
ssr3_val = GetSSR(GV3[1], valData[1])
ssr4_val = GetSSR(GV4[1], valData[1])
ssr5_val = GetSSR(GV5[1], valData[1])

print("SSR values: ")
print("ssr1 (train/E_in and val/E_val):  "+str(ssr1_train)+"  "+str(ssr1_val))
print("ssr2 (train/E_in and val/E_val):  "+str(ssr2_train)+"  "+str(ssr2_val))
print("ssr3 (train/E_in and val/E_val):  "+str(ssr3_train)+"  "+str(ssr3_val))
print("ssr4 (train/E_in and val/E_val):  "+str(ssr4_train)+"  "+str(ssr4_val))
print("ssr5 (train/E_in and val/E_val):  "+str(ssr5_train)+"  "+str(ssr5_val))

#plot the SSR values (y-values), by degree (x axis)
ssr_training = [ssr1_train, ssr2_train, ssr3_train, ssr4_train, ssr5_train]
#ssr_training = [math.sqrt(v) for v in ssr_training]
ssr_validation = [ssr1_val, ssr2_val, ssr3_val, ssr4_val, ssr5_val]
#ssr_validation = [math.sqrt(v) for v in ssr_validation]
xs = [1,2,3,4,5]
#plot both errors
plt.plot(xs, ssr_training, "g-", label="Ein")
plt.plot(xs, ssr_validation, "b-",label="Eval")
plt.show()
plt.savefig("SSRplot.png")
plt.clf()

#now plot the optimal degree polynomial (3, by observation), but trained on the entire dataset
coefs_opt, res3_opt, rank_opt, sv3_opt, rcond_opt = np.polyfit(completeData[0], completeData[1], 3, full = True)
G3_Opt = GeneratePolyData(coefs_opt, completeData[0])

#print("xseq:\n"+str(G3_Opt[0])+"\n"+str(completeData[0]))
ssr3_opt = GetSSR(G3_Opt[1], completeData[1]) #get the Ein value for this polynomial, aka ssr_residuals
print("ssr3: "+str(ssr3_opt)+"   res3: "+str(res3_opt))
ssr_actual = GetSSR(trueY, completeData[1]) #get the sum-of-square residuals, comparing the test samples with the target function
plt.plot(G3_Opt[0], G3_Opt[1], "b-", label="optimal, deg-3")
y_bar = float(sum(completeData[1])) / float(len(completeData[1]))
y_bars = [y_bar for i in range(0,len(completeData[1]))]
ss_tot = GetSSR(completeData[1], y_bars) #get SS_total from samples and their mean
coefDet = 1.0 - (ssr3_opt / ss_tot)  # R**2 is defined as 1 - SS_res / SS_tot

#plot the noisy outputs of the true function
ScatterplotDataset(completeData[0],completeData[1],"*")
plt.show()
plt.savefig("OptimalPlot.png")
print("Minimum ssr on deg-3 polynomial, using full dataset: "+str(ssr3_opt))
#coefDet = ssr3_opt / ssr_actual
print("Coefficient of determination: "+str(coefDet))
print("ss_tot: "+str(ss_tot))



















