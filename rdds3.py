import os, sys
import unicodedata
import re
import boto3
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import Row
import argparse

# needed as StreamingBody doesn't implement
# readline API
def line_iter(filelike):
    if False:
        for l in filelike.readline():
            yield l
    else:
        char_stream=''
        chunk = 1024
        done = False
        while not done:
            chars = filelike.read(chunk)
            if len(chars) < chunk:
                # break out of loop after processing
                done = True

            # add to current buffer
            char_stream += chars
            # and return all lines in the
            # current buffer
            while True :
                end = char_stream.find('\n')
                if end != -1:
                    yield char_stream[:end]
                    char_stream = char_stream[end+1:]
                else:
                    break

        # return any remaining chars as is
        if len(char_stream) > 0:
            yield char_stream

def mks3():
    access_key_id = ''
    access_key = ''
    s3 = boto3.resource('s3',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=access_key)
    return s3

def normalize(line):
    #l = line.decode('utf-8')
    l = unicodedata.normalize("NFKD", line).encode('utf-8','ignore')
    
    return l

def format_key(key) :
    if key is None:
        return 'UNKNOWN_KEY'
    else:
        key = re.sub('[^0-9a-zA-Z ]+','',key)
        key = re.sub("\d +", "", key)
        key = key.lower()
        key = key.replace(' ','_')

    return key

def getreplacements():
    replacements = set([('methods', 'materials and methods'),
                        ('material and methods','materials and methods'),
                        ('materials','materials and methods'), 
                        ('patients and methods','materials and methods'), 
                        ('subjects and methods','materials and methods'), 
                        ('methods','materials and methods'), 
                        ('methodology','materials and methods'),
                        ('method','materials and methods'),
                        ('conclusions','conclusion'),
                        ('statistical analyses','statistical analysis'),
                        ('statistics','statistical analysis'),
                        ('statistical methods','statistical analysis'),
                        ('analysis','statistical analysis'),
                        ('supporting information','supplementary material'),
                        ('supplementary data','supplementary material'),
                        ('electronic supplementary material','supplementary material'),
                        ('author contributions','authors contributions'),
                        ('authors information','authors contributions'),
                        ('list of abbreviations','abbreviations'),
                        ('ethics','ethics statement'),
                        ('ethical considerations','ethics statement'),
                        ('ethical approval','ethics statement'),
                        ('case presentation','case report'),
                        ('subjects','participants'),
                        ('patients','participants'),
                        ('study population','participants'),
                        ('patient characteristics','participants'),
                        ('mice','animals'),
                        ('design','study design'),
                        ('experimental design','study design'),
                        ('experimental section','study design'),
                        ('experimental','study design'),
                        ('research design and methods','study design'),
                        ('limitations','strengths and limitations'),
                        ('western blotting','western blot analysis'),
                        ('western blot','western blot analysis'),
                        ('experimental procedures','procedure'),
                        ('outcome measures','measures'),
                        ('remarks','concluding remarks')])
    return replacements

g_common = None
def getcommonheadings(s3, bucket):
    global g_common
    if g_common is None :
        f1 = s3.Object(bucket, 'common.txt').get()
        if 'Body' in f1:
            g_common = set()
            for l in line_iter(f1['Body']):
                g_common.add(format_key(l.strip(' \n')))
    return g_common

def getdictkeys(s3, bucket):
    s = getcommonheadings(s3, bucket)
    s.update(['generalcolumn'])
    s.update(['PMID_KEY'])
    return s

def replace_common(heading, common_headings):
    if heading is None:
        return 'generalcolumn'

    heading = format_key(heading)

    for c in common_headings:
        if c==heading:
            return c

    for r in getreplacements():
        if r[0]==heading:
            return format_key(r[1].lower())

    return 'generalcolumn'

def append_to_dict(d, k, v) :
    text = ''
    if k in d :
        text = d[k] + " "
    d[k] = text + v
    return d

# optimization: instead of sending one key per parse
# call, send multiple keys. this allows reuse of the s3
# object within a parse call
def parse_s3(s3, bucket, key, common_headings, defaultdict=None):
    if defaultdict is None:
        d = dict()
    else :
        d = defaultdict

    f1 = s3.Object(bucket, key).get()
    if 'Body' not in f1:
        return d

    heading = None
    section = ''
    section_start = None
    line = 0
    for l in line_iter(f1['Body']):
        l = l.strip(" \n")
        if len(l) == 0:
            continue

        if len(l.split(' ')) < 4  and l.endswith('.') == False:
            # possible next heading
            if len(section) > 0:
                # Apply tranformations
                heading = replace_common(heading, common_headings)
                d = append_to_dict(d, heading, section)
                heading = None
                section = ''
                section_start = None

            heading = l
            continue

        if section_start is None:
            section_start = line

        section += l
        line += 1

    # save last section as well
    if heading is not None:
        if len(section) > 0:
            heading = replace_common(heading, common_headings)
            d = append_to_dict(d, heading, section)
    
    return d

def mkf2id(s3, bucket, full_text_key):
    f1 = s3.Object(bucket, full_text_key).get()
    if 'Body' not in f1:
        return {}

    f2id = dict()
    for l in line_iter(f1['Body']):
        tokens = l.strip(' \t\n').split(' ')
        filename = tokens[-1].replace('.tar.gz', '.txt')
        uniq_id = tokens[0].strip(' \t')
        f2id[filename] = uniq_id

    return f2id

def mkrdds3(bucket, sc):
    s3 = mks3()
    f2id = mkf2id(s3, bucket, 'fulltext_id.txt')
    dict_keys = getdictkeys(s3, bucket)
    ch = getcommonheadings(s3, bucket)

    b = s3.Bucket(bucket)
    keys_rdd = sc.parallelize([i.key for i in b.objects.all()])

    def valid_fulltext(key):
        k = key.split('/')
        return k[0] == 'fulltext' and key.endswith('txt')

    # hash is used in partitioning
    keys_rdd = keys_rdd.filter(valid_fulltext).map(lambda x: (x, hash(x)%7)).partitionBy(7).glom()

    def row_maker(key_list):
        row_list=[]
        s3 = mks3()
        for k in key_list :
            d = parse_s3(s3, bucket, k[0], common_headings=ch,
                         defaultdict=dict.fromkeys(dict_keys, ''))
            fname = k[0].split('/')[-1]
            d['PMID_KEY'] = f2id[fname]
            row_list.append(Row(**d))
        return row_list

    return keys_rdd.flatMap(row_maker)

steps= """
# on the master node
my_ip=`ifconfig eth0 | grep "inet addr" | cut -d ':' -f2 | cut -d ' ' -f1`
echo $my_ip
cd /path/to/spark-2.0.1-bin-hadoop2.7/
export PATH=`pwd`/bin:`pwd`/sbin:$PATH
export MASTER_URL=spark://$my_ip:7077
sudo ./sbin/start-master.sh -h $MASTER_URL
# if on a slave node, http://www.thecloudavenue.com/2012/01/how-to-setup-password-less-ssh-to.html
# might be needed to enable password-less transfer of files/access from master,
# skip if slave is on the same machine as master or executing on EMR
#
# execute on the slave node (or execute from the same shell on master node)
sudo ./sbin/start-slave.sh <MASTER_URL> #if executing from slave node, copy master_url

# check http://http://<my_ip>:8080/ for spark status
# execute from any machine which has spark-submit and access to the file docscan.py
# must know value of MASTER_URL.
spark-submit docscans3.py <MASTER_URL> hgeorge-dropbox # will execute code on slave
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('master_url', help='master IP:PORT', nargs=1)
    parser.add_argument('bucket', help='bucket', nargs=1)
    pargs = parser.parse_args()
    conf = SparkConf().setAppName('appName').setMaster(pargs.master_url[0])
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    rdd = mkrdds3(pargs.bucket[0], sc)
    sdf = sqlContext.createDataFrame(rdd)
    print sdf[['PMID_KEY','introduction']].show(5)

if __name__ == "__main__":
    main()
