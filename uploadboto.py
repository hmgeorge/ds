import sys
import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key

access_key_id = ''
access_key = ''

if __name__ == "__main__":
    conn = S3Connection(access_key_id, access_key)
    b = conn.get_bucket('hgeorge-private')
    def _upload(key):
        with open(key) as f:
            k = Key(b)
            k.key = 'boto/'+key
            k.set_contents_from_file(f)
            b.set_acl('authenticated-read', k.key)

    _upload('fulltext_id.txt')
    _upload('common.txt')
    seek_pos = 0
    if os.access('.savedstate', os.F_OK) :
        with open('.savedstate') as f:
            seek_pos=int(f.read())

    with open('fulltext_id.txt') as f:
        f.seek(seek_pos)
        for l in f:
            fname = l.split(' ')[-1].replace('tar.gz','txt').strip(' \n')
            try :
                _upload(os.path.join('fulltext', fname))
            except:
                sys.stderr.write("skipped: line %d %s\n" % (line, fname))

            with open('.savedstate', 'w') as f1:
                f1.write(int(f.tell()))

    os.remove('.savedstate')
