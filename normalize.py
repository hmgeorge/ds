import unicodedata
def normalize(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		for i in range(len(lines)):
			l = lines[i].decode('utf-8')
			lines[i] = unicodedata.normalize("NFKD", l).encode('ascii', 'ignore')

	return lines

