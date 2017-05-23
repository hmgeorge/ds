from bs4 import BeautifulSoup
import unicodedata
import urllib2
import string
depts = ['CSE', 'COMPSCI', 'I&C SCI', 'EECS', 'MATH', 'STATS', 'IN4MATX', 'BIO SCI'] 

def parse(l, cursor, stop_char='.'):
        prereqs = ''
        word = ''
        state=0
        ignore_next_space=False
        sep=';'
        def append(p, sep, k):
                if p == '':
                        return k
                else:
                        return p + sep + k

        while cursor <= (len(l) - 1):
                c = l[cursor]
                if ignore_next_space:
                        ignore_next_space=False
                        if c == ' ':
                                cursor += 1
                                continue

                if c == ' ': #string.whitespace
                        if state==1: #STATE_DEPT
                                prereqs = append(prereqs, sep, dept + " " + word.strip(',.'))
                                state=0
                                dept=''
                        elif word == 'and':
                                sep=';'
                        elif word == 'or':
                                sep='|'
                        #some unnecessary word
                        word = ''
                elif c == '(':
                        sub_list, cursor = parse(l, cursor+1, ')')
                        prereqs = append(prereqs, sep, sub_list)
                        word = ''
                elif c == stop_char:
                        if state==1: #STATE_DEPT
                                prereqs=append(prereqs, sep, dept + " " + word.strip(',.'))
                        break
                else:
                        word += c
                        if word in ['I&C', 'BIO']:
                                word += ' '
                                ignore_next_space=True

                        if word in depts:
                                state = 1
                                dept = word
                                word = ''
                                ignore_next_space=True
                cursor += 1
        return prereqs, cursor

def mkpreqfrom(l):
        p = l.find('Prerequisite')
        c = l.find('Corequisite')
        begin = max(p+len('Prerequisite'),c+len('Corequisite'))
        preq, _ =parse(l.strip(' \n'), begin+2)
        if len(preq) == 0:
                return l[begin+1:].strip(' .\n')
        return preq

def mktitle(title):
        parts = title.split('.')
        t = ''
        for i in range(2):
                t = t + unicodedata.normalize("NFKD", parts[i].strip(' .')) + " "
        return t

def get_html():
#        response = urllib2.urlopen('http://catalogue.uci.edu/donaldbrenschoolofinformationandcomputersciences/departmentofinformatics/#courseinventory')
        response = urllib2.urlopen('http://catalogue.uci.edu/donaldbrenschoolofinformationandcomputersciences/departmentofcomputerscience/#courseinventory')
#        response = urllib2.urlopen('http://catalogue.uci.edu/donaldbrenschoolofinformationandcomputersciences/departmentofstatistics/#courseinventory')
        return response.read()
#        return open('/tmp/uci.html')

def xmain():
        doc=BeautifulSoup(get_html(), "lxml")
        courses=doc.findAll('div', class_="courseblock")
        i = 0
        for c in courses:
		title = c.findAll('p', class_="courseblocktitle")
		title_text = mktitle(title[0].text)
		ps = c.findAll('p')
                prereq = None
                keywords = ['Corequisite:', 'Prerequisite:']
                for p in ps:
                        if [p.getText().find(k) for k in keywords] != [-1]*len(keywords):
                                prereq = p
                                break

                if prereq is None:
                        print i, title_text, "=>", "N/A"
                        i += 1
                        continue

                prereq_text=unicodedata.normalize("NFKD", prereq.getText())
                req_lines = prereq_text.split('\n')

                req_lines = ''.join(req_lines);

                l = req_lines.strip(' \n')

                text = mkpreqfrom(l)

                print i, title_text, "=>", text
                i += 1

if __name__ == "__main__":
        xmain()

