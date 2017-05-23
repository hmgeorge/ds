import sys
import copy
from collections import defaultdict
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing
import argparse

def mkanons3():
    s3 = boto3.resource('s3')
    s3.meta.client.meta.events.register('choose-signer.s3.*',
                                        disable_signing)
    return s3

def dictkeys():
    keys = ['A', 'A.1', 'A.2', 'B.2', 'B', 'C.2', 'C.1']
    return keys

# optimization: Instead of a single key, a list of keys
# can be passed. this way a single s3 client can be reused.
# Also, what's mapPartitions?
def parse_s3(bucket, key, defaultdict=None):
    heading = None
    section = ''
    section_start = None
    line = 0
    if defaultdict is None:
        d = dict()
    else:
        d = defaultdict
    s3 = mkanons3()
    f1 = s3.Object(bucket, key).get()
    if 'Body' not in f1:
        return d

    lines = f1['Body'].read().splitlines() #optimize
    for l in lines:
        l = l.strip(" \n")
        if len(l) == 0:
            continue
        if len(l) < 4:
            # possible next heading
            if len(section) > 0:
                d[heading] = section
                heading = None
                section = ''
                section_start = None

            heading = l
            continue
        if section_start is None:
            section_start = line
        section += l
        line += 1

    doc_count = 0
    for k in d.keys():
        c = d[k].count('VEGF')*1.0
        d['count_VEGF_'+k] = c
        doc_count += c

    d['count_VEGF_doc'] = doc_count
    return d

def mkrdds3(bucket, sc):
    s3 = mkanons3()
    b = s3.Bucket(bucket)
    rdd = sc.parallelize([i.key for i in b.objects.all()])
    def row_maker(key):
        sys.stderr.write("DOCSCAN: row maker %s\n" % (key))
        return Row(**parse_s3(bucket, key, defaultdict=dict.fromkeys(dictkeys(), '')))
    
    return rdd.map(row_maker)

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
def xmain():
    parser = argparse.ArgumentParser()
    parser.add_argument('master_url', help='master IP:PORT', nargs=1)
    parser.add_argument('bucket', help='bucket', nargs=1)
    pargs = parser.parse_args()
    conf = SparkConf().setAppName('appName').setMaster(pargs.master_url[0])
    sc = SparkContext(conf=conf)
    rdd = mkrdds3(pargs.bucket[0], sc)
    print rdd.collect()

if __name__ == "__main__":
    xmain()
