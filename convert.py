"""
Just a short python script for converting the test data file "one-five-test.csv" in its
current format to something consumable by linear regression.

Output cols are formatted as <intensity, symmetry, +/-1>

If "yes" in 5 col: output y is +1
If "yes" in 1 col: output y is -1

Only the two input columns are used as the x vector. Prepend these with one on your own.
"""

ifile = open("one-five-test.csv","r")
ofile = open("dummy.csv","w+")
i = 0
for line in ifile.readlines():
	cols = [col.lower().strip().replace("\"","") for col in line.split(",")]
	inty = cols[-2]
	symm = cols[-1]
	isFive = False
	isOne = False
	#get the class membership bool
	if cols[-3] == "yes":
		isFive = True
	if cols[-4] == "yes":
		isOne = True
	#should be unreachable
	if isFive and isOne:
		print("WARNING isFive and isOne true for "+str(i)+" at row: "+line)
	elif not isFive and not isOne:
		print("WARNING either isFive nor isOne true for "+str(i)+" at row: "+line)
	else:
		if isFive:
			y = "1.0"
		if isOne:
			y = "-1.0"
		ofile.write(inty+","+symm+","+y+"\n")
	i += 1
		
ofile.close()
ifile.close()

