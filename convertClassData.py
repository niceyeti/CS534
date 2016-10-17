"""
Just a small script for converting the class' one-five-424.csv file to something
consumable.

File is formatted as:
	Name,Class,Intensity,Symmetry
	Row   17,1,0.159934,-1.742
	Row   33,5,0.32484,-6.360563

Mapping 1->-1 and 5->+1, the output will be:
	0.159934,-1.742,-1
	0.32484,-6.360563,1
With the first two columns representing inputs, the last column the class label (1 or -1).
"""

ifile = open("data/one-five-424.csv","r")
ofile = open("data/two-class.csv","w+")
i = 0
lines = ifile.readlines()
while i < len(lines):
	line = lines[i].strip()
	if i > 0 and len(line) > 0: #skip the first line, the csv header
		#records are formatted "Row   52,5,0.3721,-6.381437"
		cols = line.split(",")
		inty = cols[-2]
		symm = cols[-1]
		isFive = False
		isOne = False
		#get the class membership bool
		if cols[-3] == "5":
			isFive = True
		elif cols[-3] == "1":
			isOne = True
		else:
			print("ERROR class label not found: "+cols[-3])
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

