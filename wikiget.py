import urllib2
import urllib
import sys
import os
import json
import mwparserfromhell
import unicodedata
import time
import csv

"""Extract movie names and some meta for each movie
from wikipedia. Create a dir named mdb/ where
this script is executed before running"""

DEBUG = 0

def _fetch(args={}):
    url_base = 'https://en.wikipedia.org/w/api.php'
    url_args = urllib.urlencode(args)
    if DEBUG:
        print '%s\n' % (url_base + '?' + url_args)
    res = urllib2.urlopen(url_base + '?' + url_args)
    if res.code != 200:
        return json.loads('{}')

    return json.loads(res.read())
 
def fetch_titles(cname, limit=-1, output='json') :
    titles = open('mdb/titles.txt', 'w')
    skipped_titles = open('mdb/skipped_titles.txt', 'w')
    done = False
    args = {}
    args['action'] = 'query'
    args['format'] = 'json'
    args['list'] = 'categorymembers'
    args['cmtitle'] = cname
    args['cmlimit'] = '500'
    if limit != -1 :
        if limit <= 500 :
            args['cmlimit'] = str(limit)

    fetched=0
    continue_args = {}
    csv_writer = None
    csv_file = open('mdb.csv', 'w+')
    while not done :
        try :
            for k in continue_args.keys():
                args[k] = continue_args[k]
            c = _fetch(args)
            members = c['query']['categorymembers']
        except :
            sys.stderr.write("_fetch failed after %d entries\n" % (fetched))
            break

        for m in members :
            title = unicode_to_utf8(m['title'])
            try :
                titles.write(title+'\n')
                # TODO - parse will fail for films without an infobox, e.g
                # 8:46
                meta = parse_infobox(m['pageid'], title)
                sys.stderr.write('P: %s\n' % (title))
                if output == 'json':
                    meta['title'] = title
                    meta['pageid'] = m['pageid']
                    f_str = '%010d' % (fetched)
                    with open('mdb/m'+f_str+'.json', 'w') as f :
                        json.dump(meta, f)
                elif output == 'csv':
                    meta['title'] = [title]
                    meta['pageid'] = [m['pageid']]
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file,
                                                    fieldnames=sorted(meta.keys()),
                                                    restval='?')
                        csv_writer.writeheader()
                    for k in sorted(meta.keys()):
                        if len(meta[k]) > 1:
                            meta[k] = ', '.join(meta[k])
                        elif len(meta[k]) == 1:
                            meta[k] = meta[k][0]
                        else:
                            meta[k] = '?'

                    csv_writer.writerow(meta)
                else:
                    sys.stderr.write("unknown output format\n");
                    sys.exit(-1)
                        
                fetched += 1
                time.sleep(1) # etiquette
            except:
                skipped_titles.write(title+','+str(m['pageid'])+'\n')
                sys.stderr.write('S: %s\n' % (title))
                fetched += 1
                continue

        if 'continue' in c :
            continue_args = c['continue']
        else :
            continue_args = {}
            done = True

        if limit != -1 :
            if fetched >= limit:
                done = True

    titles.close()
    skipped_titles.close()
    csv_file.close()
    return fetched

#@unused
def extract_names(l):
    cursor = 0
    names = []
    while cursor < len(l) :
        begin=l[cursor:].find('[[')
        if begin == -1:
            break
            
        start = cursor+begin+2
        match = l[start:].find(']]')
        if match == -1:
            break

        names.append(l[start:start+match].strip(' \n'))
        cursor = start+match+2
    return names

def unicode_to_utf8(u_text):
    return unicodedata.normalize("NFKD", u_text).encode('utf-8', 'ignore')

"""mw node can have nested nodes, recurse until we reach
   a text or wikilink node"""
def parse_node(n):
    #print type(n)
    if type(n) == mwparserfromhell.nodes.text.Text:
        u_text = unicode_to_utf8(n.value.strip(' \n'))
        return [u_text] if len(u_text) > 1 else []
    elif type(n) == mwparserfromhell.nodes.tag.Tag:
        return []
    elif type(n) == mwparserfromhell.nodes.wikilink.Wikilink:
        _wc = n.title if n.text is None else n.text
        return parse_node(_wc.nodes[0])
    elif type(n) == mwparserfromhell.nodes.template.Template:
        l = []
        for nn in n.params :
            for ll in parse_node(nn) :
                l.append(ll)
        return l
    elif type(n) == mwparserfromhell.nodes.extras.parameter.Parameter:
        l = []
        for nn in n.value.nodes:
            for ll in parse_node(nn) :
                l.append(ll)
        return l

    return []

def parse_infobox(pageid, title):
    args = {}
    args['action'] = 'parse'
    args['format'] = 'json'
    args['prop'] = 'wikitext'
    if pageid == -1:
        args['page'] = title
    else:
        args['pageid'] = pageid

    if DEBUG:
        print args
    d = _fetch(args)
    wikitext = d['parse']['wikitext']['*']
    wc = mwparserfromhell.parse(wikitext)
    templates = wc.filter_templates(matches=lambda temp: temp.name.lower().startswith('infobox'))
    infobox = templates[0] #will raise exception. caught in caller
    meta = dict()
    meta_keys = ['director', 'producer',
                 'screenplay', 'starring',
                 'cinematography', 'editing',
                 'studio', 'distributor', 'runtime',
                 'music']

    bling_meta_keys = ['budget', 'gross']

    for k in meta_keys :
        meta[k] = []

    for k in bling_meta_keys:
        meta[k] = []

    # parse infobox
    # TODO - using meta_keys to find out presence
    # in infobox has a problem that
    # check will fail if the param in the infobox
    # isn't lower case
    for k in meta_keys:
        if infobox.has(k):
            m = []
            for n in infobox.get(k).value.nodes:
                for p in parse_node(n):
                    m.append(p)
            meta[k] = m

    for k in bling_meta_keys :
        if infobox.has(k):
            bling_matcher = lambda t : t.strip(' \n').lower().startswith('$')
            bling = infobox.get(k).value.filter_text(matches=bling_matcher)
            if len(bling) > 0:
                meta[k] = parse_node(bling[0])

    # special processing for 'released'
    year = [0]
    if infobox.has('released') :
        m = []
        for n in infobox.get('released').value.nodes:
            for p in parse_node(n):
                m.append(p)
        if len(m) != 0:
            year = [m[0]]
    meta['year'] = year
    if DEBUG :
        sys.stderr.write("parse complete for %s\n" % (title))
    return meta

def main():
    #('Category:American_films')
    #('Category:Districts_of_Kerala') #('Category:English-language_film')
    print fetch_titles('Category:American_films', limit=15, output='csv')

if __name__ == '__main__':
    main()

DEBUG_WTEXT="""
{{ubl|[[Carmen Electra]]|[[Charlie O'Connell]]|[[Brooke Hogan]]|[[Christina Bach]]|David Gallegos|Corinne Nobili}}

[[Michiel Huisman]]<br />[[Teresa Palmer]]<br />[[Sam Reid (actor)|Sam Reid]]

wc=mw.utils.parse_anything("Jimmy Smallhorne")

{{Plainlist |
* [[Chris Rock]]
* Julie Delpy
* Albert Delpy
* Alexia Landeau
* Alex Nahon
* [[Dylan Baker]]
* [[Kate Burton (actress)|Kate Burton]]}}

[[Pete Brock]]
"""
