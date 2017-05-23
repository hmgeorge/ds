import boto3
import sys
import os
import json

def mkdyndb():
    access_key_id = ''
    access_key = ''
    region = 'us-west-1'
    db = boto3.resource('dynamodb',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=access_key,
                        region_name=region
              )
    return db

def mktable(db, table_name):
    args = dict()
    args['TableName'] = table_name
    args['KeySchema'] = [ {'AttributeName': 'title',
                           'KeyType': 'HASH'},
                          {'AttributeName': 'pageid',
                           'KeyType': 'RANGE'} ]
    args['AttributeDefinitions'] = [ {'AttributeName': 'title',
                                      'AttributeType': 'S'},
                                     {'AttributeName': 'pageid',
                                      'AttributeType': 'N'} ]

    args['ProvisionedThroughput'] = {'ReadCapacityUnits': 5,
                                     'WriteCapacityUnits': 5
                                 }
    table = db.create_table(**args)
    table.meta.client.get_waiter('table_exists').wait(TableName='hgeorge-mdb')
    print table.item_count
    return table

def getormaketable(db, table):
    all_tables = db.tables.all()
    for t in all_tables:
        if t.name == table:
            return t

    return mktable(db, table)

def main():
    db = mkdyndb()
    table = getormaketable(db, 'hgeorge-mdb')
    count=10
    for fname in os.listdir('.'):
            if fname.endswith('.json'):
                with open(fname) as f:
                    d=json.load(f)
                    print 'adding item %s' % d['title']
                    table.put_item(TableName='hgeorge-mdb',
                                   Item=d)
            count -= 1
            if not count:
                break

if __name__ == "__main__":
    main()
                
