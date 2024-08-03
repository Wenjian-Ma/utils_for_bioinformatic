import sys
import re


out_file = '/media/ST-18T/Ma/HIF2GO/data/revision/1.2/go_list.txt'

out = open(out_file, 'w')
out.write('GO	name	namespace	def	level	GOs\n')

obo_file = '/media/ST-18T/Ma/HIF2GO/data/revision/1.2/go-basic.obo'
go2parent = {}
go2name = {}
go2namespace = {}
go2def = {}
data = open(obo_file, 'r').read()
Terms = data.split('[Term]')
go2level = {}
All_lines = []

for term in Terms:
    if re.search(r'\nid: GO', term) and not re.search(r'is_obsolete: true', term):
        GO_id = re.search(r'\nid: (GO:\d*)', term).groups()[0]
        name = re.search(r'name: (.*?)\n', term).groups()[0]
        go2name[GO_id] = name
        namespace = re.search(r'namespace: (.*?)\n', term).groups()[0]
        go2namespace[GO_id] = namespace
        def_inf = re.search(r'def: (.*?)\n', term).groups()[0]
        go2def[GO_id] = def_inf
        if re.search(r'is_a: (.*?) ', term):
            go_parents = re.findall(r'is_a: (GO.*?) ', term)
            go2parent[GO_id] = ';'.join(go_parents)

if 1 == 1:
    def get_parents(lines):
        for line in lines:
            go = line.split('+')[-1]
            if go in go2parent.keys():
                parents = go2parent[go]
                for parent in parents.split(';'):
                    line_new = '%s+%s' % (line, parent)
                    lines.append(line_new)
                lines.remove(line)
        return lines


    for go in go2name.keys():
        go_level = []
        lines = ['%s' % go, ]
        i = 1
        while i < 20:
            i = i + 1
            lines = get_parents(lines)
        for line in lines:
            level = len(line.split('+'))
            go_level.append(level)
        go_level = max(go_level)
        go2level[go] = go_level
        All_lines += lines

    for i in range(2, 20):
        for go in go2name.keys():
            if go2level[go] is i:
                gos = set([line.split('+')[-1] for line in All_lines if go in line])
                out.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
                go, go2name[go], go2namespace[go], go2def[go], go2level[go], ';'.join(gos)))
                out.flush()

out.close()
