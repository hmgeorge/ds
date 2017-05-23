import sys
import boto3
import os

access_key_id = ''
access_key = ''

if __name__ == "__main__":
    s3 = boto3.resource('s3',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=access_key)
    def _upload(key):
        with open(key) as f:
            o = s3.Object('hgeorge-private', key)
            o.put(ACL='authenticated-read', Body = f)

    _upload('fulltext_id.txt')
    _upload('common.txt')
    for fname in os.listdir('fulltext'):
        _upload(os.path.join('fulltext', fname))
            
