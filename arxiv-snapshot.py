with open('arxiv-metadata-oai-snapshot.json', 'r') as arxiv: 
    snapshot = []
    for i in range(50000):
        snapshot += arxiv.readline()
    with open('arxiv-snapshot', 'x') as arxiv_new:
        for line in snapshot: 
            arxiv_new.write(line) 
   